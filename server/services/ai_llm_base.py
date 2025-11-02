"""
LLM共通のベースクラスとプロンプト定数群。

目的:
- PvPエンジン上で動作するLLMボットの共通処理を集約し、
  具体的なベンダ実装（OpenAI 等）はサブクラスで最小限の実装にする。

提供機能:
- `LLMBase` 抽象クラス: `think()`の基本フロー、状態要約、メッセージ構築、
  JSON→`PlayerOrders`の変換、履歴管理、デバッグ出力。
- プロンプト定数: SYSTEM_PROMPT を構成する 3 つの部品に分割。
"""

from __future__ import annotations

import sys,os,traceback
import json
from dataclasses import dataclass, field
import time
from typing import Any, Literal, overload
from pydantic import BaseModel, Field, constr, ValidationError
from abc import ABC, abstractmethod

from server.services.ai_base import AIStat, AIThreadABC
from server.schemas import AIModel, LLMBaseConfig, MatchStatePayload, PlayerOrders, Position
from server.services.ai_llm_prompts import build_system_prompt ,match_state_payload_to_text, default_knowledge, build_summary_prompt
from server.services.ai_llm_prompts import ResponseModel, ResponseModelJA

from typing import TypeVar, Type


# =========================
# プロンプト定数
# =========================
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_AI = "assistant"

class LLMTokenUsage:
    def __init__(self, *, prompt:int=0, cache_read:int=0, completion:int=0, reasoning:int=0, cache_write:int=0):
        self.prompt_tokens = int(prompt)
        self.completion_tokens = int(completion)
        self.reasoning_tokens = int(reasoning)
        self.cache_read_tokens = int(cache_read)
        self.cache_write_tokens = int(cache_write)
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other):
        if not isinstance(other, LLMTokenUsage):
            return NotImplemented
        return LLMTokenUsage(
            prompt=self.prompt_tokens + other.prompt_tokens,
            cache_read=self.cache_read_tokens + other.cache_read_tokens,
            cache_write=self.cache_write_tokens + other.cache_write_tokens,
            completion=self.completion_tokens + other.completion_tokens,
            reasoning=self.reasoning_tokens + other.reasoning_tokens,
        )

    def __sub__(self, other):
        if not isinstance(other, LLMTokenUsage):
            return NotImplemented
        return LLMTokenUsage(
            prompt=self.prompt_tokens - other.prompt_tokens,
            cache_read=self.cache_read_tokens - other.cache_read_tokens,
            cache_write=self.cache_write_tokens - other.cache_write_tokens,
            completion=self.completion_tokens - other.completion_tokens,
            reasoning=self.reasoning_tokens - other.reasoning_tokens,
        )
    def __iadd__(self, other):
        if not isinstance(other, LLMTokenUsage):
            return NotImplemented
        self.prompt_tokens += other.prompt_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    def __isub__(self, other):
        if not isinstance(other, LLMTokenUsage):
            return NotImplemented
        self.prompt_tokens -= other.prompt_tokens
        self.cache_read_tokens -= other.cache_read_tokens
        self.cache_write_tokens -= other.cache_write_tokens
        self.completion_tokens -= other.completion_tokens
        self.reasoning_tokens -= other.reasoning_tokens
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    def to_usage_dict(self, ai_model: AIModel|None) -> dict[str,float]:
        """この使用量に基づく価格を計算して返す。ai_model が None なら空辞書。"""
        ret:dict[str,float] = {
            "prompt_tokens": self.prompt_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }
        if ai_model is not None:
            Mega = 1024*1024
            total_cost = 0.0
            if ai_model.input_price:
                if ai_model.cached_price:
                    if ai_model.cache_write_price:
                        cost = round( (self.prompt_tokens - self.cache_read_tokens - self.cache_write_tokens) / Mega * ai_model.input_price, 4)
                        ret['prompt_price'] = cost
                        total_cost += cost
                        cost = round( self.cache_read_tokens / Mega * ai_model.cached_price, 4)
                        ret["cached_price"] = cost
                        total_cost += cost
                        cost = round( self.cache_write_tokens / Mega * ai_model.cache_write_price, 4)
                        ret["cache_write_price"] = cost
                    else:
                        cost = round( (self.prompt_tokens - self.cache_read_tokens) / Mega * ai_model.input_price, 4)
                        ret['prompt_price'] = cost
                        total_cost += cost
                        cost = round( self.cache_read_tokens / Mega * ai_model.cached_price, 4)
                        ret["cached_price"] = cost
                        total_cost += cost
                else:
                    cost = round( self.prompt_tokens / Mega * ai_model.input_price, 4)
                    ret['prompt_price'] = cost
                    total_cost += cost

            if ai_model.output_price:
                cost = round( self.completion_tokens / Mega * ai_model.output_price, 4)
                ret['completion_price'] = cost
                total_cost += cost
            ret['total_price'] = round(total_cost,4)
        return ret

class LLMError(Exception):
    """LLM関連の例外。"""
    def __init__(self, *args):
        super().__init__(*args)

class LLMRateLimitError(LLMError):
    """LLMのレート制限例外。"""
    def __init__(self, message: str, retry_after: int|float|None, *args):
        super().__init__(message, *args)
        self.retry_after = retry_after

class LLMNetworkError(LLMError):
    """LLMのネットワーク例外。ネットがキレてるとかホストに接続できないとか"""
    def __init__(self, message: str, *args):
        super().__init__(message, *args)

class LLMServiceError(LLMError):
    """LLMのサービス例外。サーバに接続できたけどサービスがエラーを返したとか"""
    def __init__(self, message: str, status_code: int|None=None, *args):
        super().__init__(message, *args)
        self.status_code = status_code

class LLMApiError(LLMError):
    """LLMのAPI例外。APIがエラーを返したとか"""
    def __init__(self, message: str, error_code: str|int|None=None, *args):
        super().__init__(message, *args)
        self.error_code = error_code


class RatingInfo:
    def __init__(self, time:float, tokens:int):
        self.tm:float = time
        self.tk:int = tokens

class RatingHistory:
    def __init__(self, limit_time:float=60.0):
        self.__limit_time: float = limit_time
        self.__ratings: list[RatingInfo] = []
        self.__total_tokens: int = 0

    @property
    def tokens(self) -> int:
        self.trim(now=time.time())
        return self.__total_tokens

    @property
    def count(self) -> int:
        self.trim(now=time.time())
        return len(self.__ratings)

    def trim(self, now:float) -> None:
        limit_time = now - self.__limit_time
        if len(self.__ratings)>0 and self.__ratings[0].tm < limit_time:
            self.__ratings = [r for r in self.__ratings if now - r.tm <= self.__limit_time]
            self.__total_tokens = sum(r.tk for r in self.__ratings)

    def add(self, time:float, tokens:int) -> None:
        self.__ratings.append(RatingInfo(time, tokens))

# =========================
# 共有ユーティリティ
# =========================

def _in_bounds(x: int, y: int, w: int, h: int) -> bool:
    return 0 <= x < w and 0 <= y < h


def _is_sea(grid: list[list[int]], x: int, y: int) -> bool:
    try:
        return grid[y][x] == 0
    except Exception:
        return False
INF = float("inf")
INT_MAX = int(2**31-1)

@overload
def parse_int( value, default:None=None ) -> int|None: ...

@overload
def parse_int( value, default:int ) -> int: ...

def parse_int( value, default=None ):
    try:
        if isinstance(value,int):
            return value
        if isinstance(value,float):
            return int(value)
        return int(str(value).strip())
    except Exception:
        pass
    return default


def _nearest_sea(grid: list[list[int]], x: int, y: int) -> tuple[int, int]:
    """近傍の海セルに補正（見つからなければ元の座標）。"""
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    if _in_bounds(x, y, w, h) and _is_sea(grid, x, y):
        return x, y
    for r in range(1, 7):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = x + dx, y + dy
                if _in_bounds(nx, ny, w, h) and _is_sea(grid, nx, ny):
                    return nx, ny
    return x, y


def _pos_from_json(obj: Any) -> tuple[int, int]|None:
    if isinstance(obj, dict):
        a = obj.get("x")
        b = obj.get("y")
        if a is not None and b is not None:
            try:
                x = int(a)
                y = int(b)
                return x, y
            except Exception:
                pass
    return None

# =========================
# ユーティリティ
# =========================
def exists_env( name:str ) -> bool:
    v = os.getenv(name) if name else None
    if v is not None and v.strip() != "":
        return True
    return False

def json_loads( content:str|dict|list|None ) -> dict|list|None:
    """JSONデコード。失敗したら改行をエスケープして再試行。"""
    if not isinstance(content,str):
        return content
    content = content.strip()
    while not (content[0]=='{' and content[-1]=='}' or content[0]=='[' and content[-1]==']'):
        # ```json ... ``` または ``` ... ``` で囲まれた部分を抽出
        json_start = content.find('```json')
        if json_start>=0:
            json_start = json_start+7
        else:
            json_start = content.find('```')
            if json_start>=0:
                json_start = json_start+3
        if json_start>=0:
            json_end = content.rfind("```")
            if json_end > json_start:
                content = content[json_start:json_end].strip()
                continue
        # {}で囲まれた部分を抽出
        json_start = content.find('{')
        if json_start >= 0:
            json_end = content.rfind('}')
            if json_end > json_start:
                content = content[json_start:json_end+1].strip()
                continue
        # []で囲まれた部分を抽出
        json_start = content.find('[')
        if json_start >= 0:
            json_end = content.rfind(']')
            if json_end > json_start:
                content = content[json_start:json_end+1].strip()
                continue
        break

    while True:
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            if e.msg == "Expecting ',' delimiter":
                content = content[:e.pos] + ',' + content[e.pos:]
                continue
            elif e.msg == "Extra data":
                content = content[:e.pos]
                continue
            elif content[e.pos] == '\n':
                content = content[:e.pos] + '\\n' + content[e.pos+1:]
                continue
            else:
                raise

T = TypeVar("T", bound=BaseModel)

def parse_output_to_model(content: str, *, output_format: Type[T] | None = None) -> T | None:
    try:
        if content and output_format:
            try:
                json_data = json_loads(content)
                content = json.dumps(json_data, ensure_ascii=False)
            except:
                pass
            answer_data = output_format.model_validate_json(content)
            return answer_data
    except ValidationError:
        return None
    return None

# =========================
# ベースクラス
# =========================
class HistMsg:
    def __init__(self, turn:int, role:str, content:str):
        self.turn = turn
        self.role = role
        self.content = content
    def to_dict(self) -> dict[str,str]:
        return {"role": self.role, "content": self.content}
    def __repr__(self) -> str:
        return f"HistMsg(turn={self.turn}, role={self.role}, content={self.content[:30]}...)"

class LLMBase(AIThreadABC, ABC):
    """LLMベースボット。

    サブクラスは `_ensure_client_ready()` と `_ask_llm()` を実装する。
    それ以外のフロー（状態要約→メッセージ構築→検証→再試行）は共通実装を利用。
    """

    def __init__(
        self,
        store,
        match_id: str,
        ai_model: AIModel,
        config: LLMBaseConfig|None = None,
    ) -> None:
        super().__init__(store=store, match_id=match_id)
        self._config: LLMBaseConfig = config or LLMBaseConfig()
        self.aimodel: AIModel = ai_model
        self._config.model = ai_model.model
        self.use_output_format:bool = False
        # user/assistant の履歴（systemは毎回先頭に付与）
        self._history: list[HistMsg] = []
        self._history_window_start: int = 0
        self._history_window_end: int = 0
        self.max_turn:int = 0 # マッチの最大ターン数
        self._knowledge_content: str|None = None
        # 直近ターンのAI診断（PvE向けUI用）
        self._last_ai_diag: dict[str, Any]|None = None
        self._last_usage_tokens: tuple[int, int]|None = None  # (pt, ct)
        # 直近のトークン使用量
        self._last_token_usage: LLMTokenUsage = LLMTokenUsage()
         # 累計トークン使用量
        self._total_token_usage = LLMTokenUsage()
        self.__datadir = os.path.join("tmp", f"llm_{self._config.model}")
        # Geminiレーティング管理
        self._rating_info: RatingHistory = RatingHistory(limit_time=60.0)  # 直近1分間のトークン数

    @property
    def name(self) -> str:
        return self.aimodel.name


    def set_api_key(self, api_key: str|None) -> None:
        """APIキーを設定する。"""
        if api_key and len(api_key) > 8:
                dmp = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else api_key
                print(f"Setting io.net API key to {dmp}")
                self._config.api_key = api_key
        else:
            print("Clearing io.net API key")
            self._config.api_key = None

    def get_max_input_tokens(self) -> int:
        n = min( parse_int(self.aimodel.max_input_tokens,INT_MAX),parse_int(self._config.max_input_tokens,INT_MAX) )
        if n<=INT_MAX:
            return n
        else:
            raise RuntimeError("this model does not support max_input_tokens")

    def get_max_output_tokens(self) -> int|None:
        n = min( parse_int(self.aimodel.max_output_tokens,INT_MAX), parse_int(self._config.max_output_tokens,INT_MAX) )
        if n<=INT_MAX:
            return n
        else:
            return None # raise RuntimeError("this model does not support max_output_tokens")

    def get_input_strategy(self) -> Literal['truncate', 'summarize', 'api']:
        s = self._config.input_strategy or self.aimodel.input_strategy or 'api'
        if s not in ('truncate', 'summarize', 'api'):
            s = 'api'
        return s

    def reset_token_usage(self) -> None:
        self._last_token_usage = LLMTokenUsage()
        self._total_token_usage = LLMTokenUsage()
        self._last_usage_tokens = None

    def messages_to_content(self, msgs: list[dict[str,str]] ) -> str:
        msgs_content = "\n\n".join( [ f"{m['role']}:\n{m['content']}" for m in msgs ] )
        return msgs_content

    @abstractmethod
    def count_tokens(self, msgs: list[dict[str, str]], *, output_format: type[BaseModel]|None=None) -> int:
        """メッセージのトークン数を数える。"""

    @abstractmethod
    def LLM(self, msgs: list[dict[str, str]], *, output_format: type[BaseModel]|None=None) -> str:
        """LLMへの問い合わせを実装し、文字列を返す。失敗時は例外。"""

    def _ask_summarize(self, token_over_count:int, system_prompt:str) -> None:
        """履歴が長くなったら要約して短くする。"""
        if token_over_count <= 0 or self.get_input_strategy() == 'api':
            return

        elif self.get_input_strategy() == 'summarize':
            self._dbg_print(self._last_turn, f"Summarizing history (size={self._history_window_end - self._history_window_start})...")

            msgs = [m.to_dict() for m in self._history[self._history_window_start:self._history_window_end]]
            last_turn = self._history[self._history_window_end -1].turn if self._history_window_end -1 >=0 else 0
            start_turn = last_turn -1 if last_turn -1 >=0 else 0
            if self.aimodel.cached_price and self.aimodel.cached_price>0:
                # キャッシュ効くならプロンプトも入れる
                msgs.insert(0, {"role": ROLE_SYSTEM, "content": system_prompt})

            summary_prompt = build_summary_prompt( language=self._config.language )
            request_msg = {'role': ROLE_USER, 'content': summary_prompt}
            msgs.append( request_msg )

            content = self.LLM(msgs,output_format=ResponseModel)

            if content:
                self._dbg_print(self._last_turn, f"Summary: {content}")
                self._history.insert(self._history_window_end, HistMsg(start_turn, ROLE_USER, summary_prompt))
                self._history.insert(self._history_window_end+1, HistMsg(last_turn, ROLE_AI, content))
                self._history_window_start = self._history_window_end+1
                self._history_window_end = self._history_window_start

        elif self.get_input_strategy() == 'truncate':
            self._dbg_print(self._last_turn, f"Truncating history (size={self._history_window_end - self._history_window_start})...")
            last_turn = self._history[self._history_window_end -1].turn if self._history_window_end -1 >=0 else 0
            start_turn = last_turn -3 if last_turn -3 >=0 else 0
            start_index = next((i for i, m in enumerate(self._history) if m.turn == start_turn), self._history_window_end)
            self._history_window_start = start_index

    def _ask_review(self, system_prompt: str,review_prompt:str) -> str:
        self._dbg_print(self._last_turn, "Reviewing knowledge...")
        msgs = self._build_messages()
        msgs.insert(0, {"role": ROLE_SYSTEM, "content": system_prompt})
        msgs.append({"role": ROLE_USER, "content": review_prompt})
        content = self.LLM(msgs)
        self._dbg_print(self._last_turn, f"Review: {content}")
        return content

    def startup(self) -> bool:
        self._knowledge_content = self.load_knowledge() or default_knowledge(language=self._config.language)
        return True

    def abort(self, payload:MatchStatePayload|None):
        turn = payload.turn if payload else self._last_turn
        user_msg = match_state_payload_to_text(payload) if payload else "No match data"
        self._append_history(turn, ROLE_USER, f"Aborted:\n\n {user_msg}")

    def over(self, payload:MatchStatePayload|None):
        turn = payload.turn if payload else self._last_turn
        user_msg = match_state_payload_to_text(payload) if payload else "No match data"
        self._append_history(turn, ROLE_USER, f"Over:\n\n {user_msg}")

    # --- Core ---
    def think(self, payload: MatchStatePayload) -> PlayerOrders:  # type: ignore[override]
        self.max_turn = payload.max_turn
        grid = self.maparray or payload.map or []
        if (not self.maparray) and payload.map:
            self.maparray = payload.map
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0
        thinking_count0: LLMTokenUsage = LLMTokenUsage()

        if not self._ensure_client_ready():
            self._dbg_print(payload.turn, "LLM client not ready")
            self.stat = AIStat.ERROR
            self._set_last_ai_diag(turn=payload.turn, ms=0, thinking="not available", usage=thinking_count0)
            raise RuntimeError("LLM client not ready")

        # self._dbg_print(payload.turn, f"Start thinking....")
        if self._total_token_usage:
            thinking_count0 += self._total_token_usage

        system_prompt = build_system_prompt( grid, side=self.side, max_turn=self.max_turn, language=self._config.language )
        response_model = ResponseModel if self._config.language=='en' else ResponseModelJA
        orders = None
        #state_json = self._summarize_state(payload, w, h)
        #user_msg = self._make_user_msg(state_json)
        user_msg = match_state_payload_to_text(payload)
        msgs = self._build_messages(user_msg)
        msgs.insert(0, {"role": ROLE_SYSTEM, "content": system_prompt})
        self._dbg_print(payload.turn,f"input\n{user_msg.replace('\n\n','\n')}")

        if self.get_input_strategy() != 'api':
            token_over_count = self.count_tokens(msgs, output_format=response_model) - self.get_max_input_tokens()
            if token_over_count > 0:
                self._ask_summarize( token_over_count, system_prompt)
                # 要約したら再構築
                msgs = self._build_messages(user_msg)
                msgs.insert(0, {"role": ROLE_SYSTEM, "content": system_prompt})

        # 要約用のマーキング
        self._history_window_end = len(self._history)

        t = 0; ntry = 3
        thinking_list = []
        abort_ex = None
        ms = 0
        while t < ntry:
            t += 1
            if not self.is_match_active() or not self.is_ready():
                self._dbg_print(payload.turn, f"[t:{t}] Thinking aborted")
                break
            if t>1:
                time.sleep(1.2)
                if not self.is_match_active() or not self.is_ready():
                    self._dbg_print(payload.turn, f"[t:{t}] Thinking aborted")
                    break
            self._dbg_print(payload.turn, f"[t:{t}] Thinking attempt ...")
            t0 = time.perf_counter()
            try:
                content = self.LLM(msgs, output_format=response_model)
            except LLMRateLimitError as ex:
                abort_ex = ex
                self.stat = AIStat.ERROR
                self._dbg_print(payload.turn, f"[t:{t}] Rate limit error: {ex}, retry_after={ex.retry_after}")
                raise ex
            except Exception as ex:
                abort_ex = ex
                self.stat = AIStat.ERROR
                traceback.print_exc(file=sys.stderr)
                content = f"Exception: {ex}"
            finally:
                ms += int((time.perf_counter() - t0) * 1000)
            self._dbg_print(payload.turn, f"[t:{t}] output: {content}")
            try:
                decoded = json.dumps(json_loads(content), ensure_ascii=False)
                if decoded != content:
                    r = len(decoded)/len(content)
                    if r < 0.8 or 1.2 < r:
                        self.store.add_json_errors(self.match_id,self.token)
                        self._dbg_print(payload.turn, f"[t:{t}] fixed: {content}")
                    content = decoded
            except Exception as ex:
                exmsg = str(ex)[:200]
                self._dbg_print(payload.turn, f"[t:{t}] JSON decode error: {exmsg}")
                if t==1:
                    self.store.add_json_errors(self.match_id,self.token)
                    msgs.append( {"role": ROLE_AI, "content": content} )
                    msgs.append( {"role": ROLE_USER, "content": f"JSON decode error: {exmsg}"} )
                    continue

            if abort_ex is not None:
                thinking_list.append(f"Exception: {abort_ex}")
                break

            thinking = ""
            errs = []
            orders = None
            try:
                # thinking 抽出（UI向け）
                thinking = self._extract_thinking(content)
                orders, err = self._content_to_orders(content, grid, w, h)
                if err and len(err) > 0:
                    self.store.add_json_errors(self.match_id,self.token)
                    errs.append(err)
                else:
                    errs = self.validate_orders(orders)
            except Exception as ex:
                traceback.print_exc(file=sys.stderr)
                errs.append(f"Exception: {ex}")
                t = ntry  # 強制終了

            if thinking:
                thinking_list.append(thinking)
            if errs and len(errs)>0:
                thinking_list.extend(errs)
                if ntry<=t:
                    self._dbg_print(payload.turn, f"[t:{t}] ERROR: " + "\n".join(errs))
                    thinking_list.append("Giveing up!")
                    break
                else:
                    self._dbg_print(payload.turn, f"[t:{t}] ERROR: " + "\n".join(errs))
                    thinking_list.append("Retrying...")
                    user_err = "ERROR: invalid your order, rejected: " + "\n".join(errs) + "\nPlease fix your order."
                    msgs.append( {"role": ROLE_AI, "content": content} )
                    msgs.append( {"role": ROLE_USER, "content": user_err} )
            else:
                self._dbg_print(payload.turn, f"[t:{t}] thinking done")
                break

        #
        self._append_history(payload.turn, ROLE_USER, user_msg )
        self._append_history(payload.turn, ROLE_AI, content )
        # 集計
        thinking_count = self._total_token_usage - thinking_count0
        self._dbg_print(payload.turn, f"Token usage for thinking: prompt={thinking_count.prompt_tokens}/{self._total_token_usage.prompt_tokens}, total={thinking_count.total_tokens}/{self._total_token_usage.total_tokens}")
        # モニタ用に保持（注文JSONは入れない）
        self._set_last_ai_diag(turn=payload.turn, ms=ms, thinking="\n".join(thinking_list), usage=thinking_count )
        
        if abort_ex is not None:
            self.stat = AIStat.ERROR
            raise abort_ex
        
        return orders if orders else PlayerOrders()

    def debug_call(self, user_msg:str, *, output_format:type[BaseModel]|None=None ) -> str|None:
        """デバッグ用に直接 LLM を呼び出す。失敗時は None。"""
        if not self._ensure_client_ready():
            return None

        thinking_count0: LLMTokenUsage = LLMTokenUsage()
        if self._total_token_usage:
            thinking_count0 += self._total_token_usage

        system_prompt = build_system_prompt( self.maparray, side=self.side, max_turn=self.max_turn, language=self._config.language )
        msgs = self._build_messages(user_msg)
        msgs.insert(0, {"role": ROLE_SYSTEM, "content": system_prompt})

        t = 0; ntry = 3
        thinking_list = []

        ms = 0
        while t < ntry:
            t += 1
            if not self.is_match_active() or not self.is_ready():
                break
            if t>1:
                time.sleep(1.2)
                if not self.is_match_active() or not self.is_ready():
                    break
            t0 = time.perf_counter()
            try:
                content = self.LLM(msgs, output_format=output_format)
                ms = int((time.perf_counter() - t0) * 1000)
            except Exception as ex:
                traceback.print_exc(file=sys.stderr)
                break
            finally:
                ms += int((time.perf_counter() - t0) * 1000)
            self._debug_print(9999, None, content)
            self._append_history(self._last_turn, ROLE_USER, user_msg)
            self._append_history(self._last_turn, ROLE_AI, content)

            thinking = ""
            errs = []
            try:
                # thinking 抽出（UI向け）
                thinking = self._extract_thinking(content)
            except Exception as ex:
                traceback.print_exc(file=sys.stderr)
                errs.append(f"Exception: {ex}")
                t = ntry  # 強制終了

            if thinking:
                thinking_list.append(thinking)
        return '\n'.join(thinking_list)

    def cleanup(self):
        """作戦終了後にレビューする"""
        try:
            #system_prompt = build_system_prompt( self.maparray, side=self.side, max_turn=self.max_turn, language=self._config.language )
            #review_prompt = REVIEW_PROMPT
            t0 = time.perf_counter()
            # content = self._ask_review( system_prompt, review_prompt)
            ms = int((time.perf_counter() - t0) * 1000)
            # thinking_text = self._extract_thinking(content)
            # content = thinking_text if thinking_text else content
            # self.save_knowledge(content)
        except Exception as ex:
            traceback.print_exc(file=sys.stderr)

    # --- Diag helpers ---
    def content_to_json(self, content) -> dict|list|None:
        if isinstance(content, dict) or isinstance(content, list):
            return content
        try:
            obj = json_loads(content)
            if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
                return obj[0]
            return obj
        except Exception as ex:
            return None

    def _extract_thinking(self, content: str) -> str|None:
        """LLM応答から thinking フィールドを抽出して返す。失敗時は None。"""
        try:
            obj = self.content_to_json(content)
            txt = self._unwrap_json("thinking", obj)
            if txt is not None:
                return str(txt)
        except Exception:
            pass
        return None

    def _diag_model_name(self) -> str:
        """サブクラスでモデル名を返す。"""
        if self.aimodel.reasoning:
            return f"{self.aimodel.model}({self.aimodel.reasoning})"
        return self.aimodel.model

    def _set_last_usage_tokens(self, usage: LLMTokenUsage|None) -> None:
        if usage is None:
            self._last_token_usage = LLMTokenUsage()
            return
        self._last_token_usage = usage
        self._total_token_usage += usage

    def _set_last_ai_diag(self, *, turn: int, ms: int, thinking: str|None, usage: LLMTokenUsage|None ) -> None:
        cost = None
        total_cost = None
        if usage is not None:
            usage_dict = usage.to_usage_dict(self.aimodel)
            cost = usage_dict.get('total_price', 0.0) if usage_dict else 0.0
        if self._total_token_usage:
            usage_dict = self._total_token_usage.to_usage_dict(self.aimodel)
            total_cost = usage_dict.get('total_price', 0.0) if usage_dict else 0.0
        self._last_ai_diag = {
            "turn": int(turn),
            "model": self._diag_model_name(),
            "ms": int(ms),
            "pt": usage.prompt_tokens if usage and usage.prompt_tokens else None,
            "ct": usage.completion_tokens if usage and usage.completion_tokens else None,
            "cost": float(cost) if cost else None,
            "total_cost": float(total_cost) if total_cost else None,
            "thinking": thinking if (thinking is None or isinstance(thinking, str)) else str(thinking),
        }

    # Match から参照される想定
    def get_ai_diag(self) -> dict[str, Any]|None:
        return self._last_ai_diag

    # --- Helpers ---
    def _summarize_state(self, payload: MatchStatePayload, w: int, h: int) -> dict[str, Any]:
        my = payload.units
        intel = payload.intel
        me = {
            "carrier": (None if not my.carrier else {
                "id": my.carrier.id,
                "x": my.carrier.x,
                "y": my.carrier.y,
                "hp": my.carrier.hp,
                "vision": my.carrier.vision,
                "speed": my.carrier.speed,
                "fuel": my.carrier.fuel,
            }),
            "squadrons": []
        }
        for sq in (my.squadrons or []):
            me["squadrons"].append({
                "id": sq.id,
                "state": sq.state,
                "x": sq.x,
                "y": sq.y,
                "hp": sq.hp,
            })

        enemy_hint = {
            "carrier_last": (None if not intel.carrier else {
                "id": intel.carrier.id,
                "x": intel.carrier.x,
                "y": intel.carrier.y,
            }),
            "squadrons": []
        }
        for sq in (intel.squadrons or []):
            enemy_hint["squadrons"].append({
                "id": sq.id,
                "x": sq.x,
                "y": sq.y,
            })

        return {
            "turn": payload.turn,
            "map_size": {"w": w, "h": h},
            "yours": me,
            "enemy_hint": enemy_hint,
            "notes": "Return only the JSON object requested.",
        }


    def _build_messages(self, user_msg: str|None=None) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = [m.to_dict() for m in self._history[self._history_window_start:]]
        if user_msg:
            msgs.append({"role": ROLE_USER, "content": user_msg})
        return msgs

    def _append_history(self, turn:int, role:str, content:str) -> None:
        if not isinstance(turn,int) or turn<0:
            raise ValueError("invalid turn")
        if not isinstance(role,str) or role.strip()=="":
            raise ValueError("invalid role")
        if not isinstance(content,str) or content.strip()=="":
            raise ValueError("invalid content")
        self._history.append(HistMsg(turn=turn, role=role, content=content))

    def _debug_print(self, turn: int, state_json: dict[str, Any]|None, content: str) -> None:
        llm=self._diag_model_name()
        print(f"[{llm}][turn:{turn}] LLM応答: {content}")

    def _unwrap_json(self, key:str, obj ):
        if isinstance(obj, dict):
            if key in obj:
                return obj[key]
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    child = self._unwrap_json(key, v)
                    if child is not None:
                        return child
        elif isinstance(obj, list) and len(obj)>0:
            for v in obj:
                if isinstance(v, (dict, list)):
                    child = self._unwrap_json(key, v)
                    if child is not None:
                        return child
        # elif isinstance(obj, (str, int, float, bool)):
        #     return str(obj).strip()
        return None

    def _content_to_orders(self, content: str, grid: list[list[int]], w: int, h: int) -> tuple[PlayerOrders, str]:
        try:
            obj = json_loads(content)
        except Exception as ex:
            return PlayerOrders(), f"JSON解析エラー: {ex}"
        obj = self.content_to_json(content)

        carrier_target = None
        launch_target = None
        msg = []
        carrier_target_value = self._unwrap_json("carrier_target", obj)
        if carrier_target_value is not None:
            pt = _pos_from_json(carrier_target_value)
            if pt is None:
                msg.append("invalid carrier_target format")
            else:
                x, y = pt
                carrier_target = Position(x=x, y=y)

        launch_target_value = self._unwrap_json("launch_target", obj)
        if launch_target_value is not None:
            pt = _pos_from_json(launch_target_value)
            if pt is None:
                msg.append("invalid launch target value format")
            else:
                x, y = pt
                launch_target = Position(x=x, y=y)

        return PlayerOrders(carrier_target=carrier_target, launch_target=launch_target), "\n".join(msg)

    def load_knowledge(self) -> str|None:
        """ナレッジを読み込む。"""
        file_path = os.path.join(self.__datadir, "knowledge.txt")
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    knowledge = f.read()
                knowledge = knowledge.strip()
                if knowledge and len(knowledge) > 0:
                    return knowledge
            except Exception:
                pass
        return None

    def save_knowledge(self, knowledge: str):
        """ナレッジを保存する。"""
        try:
            os.makedirs(self.__datadir, exist_ok=True)
            file_path = os.path.join(self.__datadir, "knowledge.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(knowledge)
        except Exception:
            pass
        return

    def to_usage_dict(self) -> dict[str,float]:
        return self._total_token_usage.to_usage_dict(self.aimodel)

    def save_history(self, logspath: str):
        """履歴を保存する。"""
        try:
            os.makedirs(os.path.dirname(logspath), exist_ok=True)
            with open(logspath, "w", encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return
