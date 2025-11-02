"""
OpenAIベースのLLMボット実装（PvPエンジン上で動作）。

このモジュールは OpenAI 依存部のみを提供し、共通ロジックは
`server/services/ai_llm_base.py` の `LLMBase` に集約した。
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
import time
from typing import Any, Dict, Optional, List

import httpx
from pydantic import BaseModel

# If tests or imports run this module directly, ensure project-local `config.env` is loaded so
# environment variables like OPENAI_API_KEY are available during import-time.
try:
    from pathlib import Path
    from dotenv import load_dotenv
    # ai_openai.py is at server/services; project root is two levels up
    p = Path(__file__).resolve().parents[2] / "config.env"
    if p.exists():
        load_dotenv(dotenv_path=str(p))
except Exception:
    # dotenv may be unavailable in some minimal test environments; ignore if not present
    pass

from server.services.ai_llm_base import (
    LLMError, LLMRateLimitError, LLMTokenUsage,
    LLMBase,
    LLMBaseConfig,
    ROLE_SYSTEM, ROLE_USER, ROLE_AI,
)
from server.schemas import PlayerOrders, AIModel, AIProvider, AIListResponse


# openai SDK（新API）。requirements.txtで `openai` 指定済み。
USE_OPENAI = False
try:
    from openai import OpenAI, Omit, APIConnectionError, APITimeoutError,RateLimitError
    import openai
    from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
    from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
    from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
    from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
    from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam
    from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
    from openai.types.chat.chat_completion_developer_message_param import ChatCompletionDeveloperMessageParam
    from openai.types.chat.completion_create_params import ResponseFormat
    from openai.types.responses.response_input_param import ResponseInputParam
    from openai.types.responses.easy_input_message_param import EasyInputMessageParam
    from openai.types.responses.response import Response
    from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema
    from openai.types.shared_params.response_format_json_object import ResponseFormatJSONObject
    from openai.lib._parsing import type_to_response_format_param
    USE_OPENAI = True
except Exception:  # ランタイム環境により未導入の可能性に備える
    USE_OPENAI = False


_TIKTOKEN_ENCODERS: dict[str, Any] = {}


def _get_token_encoder(model_name: str) -> Any | None:
    """tiktokenエンコーダをキャッシュ付きで取得する。"""
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None

    if model_name in _TIKTOKEN_ENCODERS:
        return _TIKTOKEN_ENCODERS[model_name]

    encoder = None
    try:
        encoder = tiktoken.encoding_for_model(model_name)
    except KeyError:
        fallbacks: tuple[str, ...]
        if model_name.startswith(("gpt-5", "o3", "o4")):
            fallbacks = ("o200k_base", "cl100k_base", "p50k_base")
        else:
            fallbacks = ("cl100k_base", "p50k_base")
        for name in fallbacks:
            try:
                encoder = tiktoken.get_encoding(name)
                if encoder is not None:
                    break
            except KeyError:
                continue
    except Exception:
        encoder = None

    _TIKTOKEN_ENCODERS[model_name] = encoder
    return encoder

OPENAI_MODELS = AIProvider(name="OpenAI", models=[
            AIModel(name="gpt-5-nano", model="gpt-5-nano", reasoning='minimal', max_input_tokens=4000000, max_output_tokens=128000, input_price=0.05, cached_price=0.005, output_price=0.4),
            AIModel(name="gpt-5-nano-L", model="gpt-5-nano", reasoning='low', max_input_tokens=4000000, max_output_tokens=128000, input_price=0.05, cached_price=0.005, output_price=0.4),
            AIModel(name="gpt-5-nano-M", model="gpt-5-nano", reasoning='medium', max_input_tokens=4000000, max_output_tokens=128000, input_price=0.05, cached_price=0.005, output_price=0.4),
            AIModel(name="gpt-5-nano-H", model="gpt-5-nano", reasoning='high', max_input_tokens=4000000, max_output_tokens=128000, input_price=0.05, cached_price=0.005, output_price=0.4),
            AIModel(name="gpt-4.1-nano", model="gpt-4.1-nano", max_input_tokens=1047576, max_output_tokens=32768, input_price=0.1, cached_price=0.025, output_price=0.4),
            AIModel(name="gpt-4o-mini", model="gpt-4o-mini", max_input_tokens=128000, max_output_tokens=16384, input_price=0.15,cached_price=0.075, output_price=0.6),
            AIModel(name="gpt-5-mini", model="gpt-5-mini", reasoning='minimal', max_input_tokens=4000000, max_output_tokens=128000, input_price=0.25, cached_price=0.025, output_price=2.0),
            AIModel(name="gpt-5-mini-L", model="gpt-5-mini", reasoning='low', max_input_tokens=4000000, max_output_tokens=128000, input_price=0.25, cached_price=0.025, output_price=2.0),
            AIModel(name="gpt-5-mini-M", model="gpt-5-mini", reasoning='medium', max_input_tokens=4000000, max_output_tokens=128000, input_price=0.25, cached_price=0.025, output_price=2.0),
            AIModel(name="gpt-5-mini-H", model="gpt-5-mini", reasoning='high', max_input_tokens=4000000, max_output_tokens=128000, input_price=0.25, cached_price=0.025, output_price=2.0),
            AIModel(name="gpt-4.1-mini", model="gpt-4.1-mini", max_input_tokens=1047576, max_output_tokens=32768, input_price=0.4, cached_price=0.1, output_price=1.6),
            AIModel(name="gpt-3.5-turbo", model="gpt-3.5-turbo", max_input_tokens=4096, max_output_tokens=4096, input_price=0.5, output_price=1.5, input_strategy='summarize', output_format="json_object"),
            AIModel(name="gpt-4-turbo", model="gpt-4-turbo", max_input_tokens=128000, max_output_tokens=4096, input_price=10, output_price=30, input_strategy='summarize', output_format="json_object"),
            AIModel(name="gpt-4", model="gpt-4", max_input_tokens=8192, max_output_tokens=8192, input_price=30, output_price=60, input_strategy='summarize', output_format="json_object"),
            AIModel(name="o4-mini", model="o4-mini", reasoning='minimal', max_input_tokens=200000, max_output_tokens=100000, input_price=1.1, cached_price=0.55, output_price=4.4),
            AIModel(name="o3-mini", model="o3-mini", reasoning='minimal', max_input_tokens=200000, max_output_tokens=100000, input_price=1.1, cached_price=0.55, output_price=4.4),
            AIModel(name="gpt-5-chat", model="gpt-5-chat-latest", max_input_tokens=128000, max_output_tokens=16000, input_price=1.25, cached_price=0.125, output_price=10.0, input_strategy='summarize', output_format='json_object'),
            AIModel(name="gpt-5", model="gpt-5", reasoning='minimal', max_input_tokens=4000000, max_output_tokens=128000, input_price=1.25, cached_price=0.125, output_price=10.0),
            AIModel(name="gpt-5-L", model="gpt-5", reasoning='low', max_input_tokens=4000000, max_output_tokens=128000, input_price=1.25, cached_price=0.125, output_price=10.0),
            AIModel(name="gpt-5-M", model="gpt-5", reasoning='medium', max_input_tokens=4000000, max_output_tokens=128000, input_price=1.25, cached_price=0.125, output_price=10.0),
            AIModel(name="gpt-5-H", model="gpt-5", reasoning='high', max_input_tokens=4000000, max_output_tokens=128000, input_price=1.25, cached_price=0.125, output_price=10.0),
            AIModel(name="gpt-5-codex", model="gpt-5-codex", reasoning=None, max_input_tokens=4000000, max_output_tokens=128000, input_price=1.25, cached_price=0.125, output_price=10.0),
            AIModel(name="gpt-5-codex-L", model="gpt-5-codex", reasoning='low', max_input_tokens=4000000, max_output_tokens=128000, input_price=1.25, cached_price=0.125, output_price=10.0),
            AIModel(name="gpt-5-codex-M", model="gpt-5-codex", reasoning='medium', max_input_tokens=4000000, max_output_tokens=128000, input_price=1.25, cached_price=0.125, output_price=10.0),
            AIModel(name="gpt-5-codex-H", model="gpt-5-codex", reasoning='high', max_input_tokens=4000000, max_output_tokens=128000, input_price=1.25, cached_price=0.125, output_price=10.0),
            AIModel(name="gpt-4.1", model="gpt-4.1", max_input_tokens=1047576, max_output_tokens=32768, input_price=2.0, cached_price=0.5, output_price=8.0),
            AIModel(name="o3", model="o3", reasoning='minimal', max_input_tokens=200000, max_output_tokens=1000000, input_price=2.0, cached_price=0.5, output_price=8.0),
            AIModel(name="gpt-4o", model="gpt-4o", max_input_tokens=128000, max_output_tokens=16384, input_price=2.5, cached_price=1.25, output_price=10.0),
        ])
        # AIModel(name="gpt-4-turbo", model="gpt-4-turbo", max_input_tokens=128000, max_output_tokens=4096)
        # AIModel(name="gpt-4", model="gpt-4", max_input_tokens=8192, max_output_tokens=8192)
        # AIModel(name="o1", model="o1", max_input_tokens=200000, max_output_tokens=1000000)
        # AIModel(name="o1-mini", model="o1-mini", max_input_tokens=128000, max_output_tokens=65536)


def is_resoning_model(model_name: str) -> bool:
    """モデル名が推論強化モデルかどうかを返す。"""
    if model_name.startswith("gpt-5") or model_name.startswith("o3") or model_name.startswith("o4"):
        return True
    return False

def to_response_format( output_model: type[BaseModel]|None, output_strict: bool = False ) -> ResponseFormat|Omit:
    if USE_OPENAI is None or not USE_OPENAI:
        return Omit()
    if output_model is None or not issubclass(output_model, BaseModel):
        return Omit()
    try:
        resp_fmt = type_to_response_format_param(output_model)
        json_schema = resp_fmt.get("json_schema", {}) if isinstance(resp_fmt, dict) else {}
        if isinstance(output_strict,bool) and isinstance(json_schema.get('strict'), bool) and json_schema.get('strict') != output_strict:
            json_schema['strict'] = output_strict
        return resp_fmt
    except Exception:
        pass
    return Omit()

@dataclass
class OpenAIConfig(LLMBaseConfig):
    model: str = field(default=OPENAI_MODELS.models[0].name)
    api_key: Optional[str] = None
    temperature: float = 0.2

class CarrierBotOpenAI(LLMBase):
    """OpenAI LLM を用いるボット（LLMBase 継承）。"""

    def __init__(
        self,
        store,
        match_id: str,
        *,
        name: str|None = None,
        ai_model: AIModel|None = None,
        config: LLMBaseConfig|None = None,
    ) -> None:
        ai_model = OPENAI_MODELS.get_model(name=name,config=config) if ai_model is None else ai_model
        assert ai_model is not None, f"Unknown OpenAI model: {name}"
        config = config or LLMBaseConfig()
        config.model = ai_model.model
        super().__init__(store=store, match_id=match_id, ai_model=ai_model, config=config)
        # サブクラス固有
        self.use_output_format:bool = True
        self._client: OpenAI|None = None
        # 会話履歴
        self._prompt_cache_key: str|None = None
        self._prev_prompt: str|None = None
        self._prev_ai_msg: str|None = None
        self._prev_response_id: str|None = None

    @staticmethod
    def get_model_names() -> AIProvider:
        return OPENAI_MODELS

    @staticmethod
    def get_model(name: str|None = None) -> AIModel|None:
        if name:
            return OPENAI_MODELS.find(name) or None
        return None

    @staticmethod
    def get_default_model() -> AIModel:
        return OPENAI_MODELS.models[0]

    # --- OpenAI依存部 ---
    def _ensure_client_ready(self) -> bool:
        if self._client is not None:
            return True
        if USE_OPENAI is None:
            return False
        if not self._config.api_key and not os.getenv("OPENAI_API_KEY"):
            return False
        try:
            to = httpx.Timeout(timeout=300, connect=30.0, pool=False)
            # base_url 指定がある場合に対応
            if self.aimodel.base_url:
                self._client = OpenAI(api_key=self._config.api_key, base_url=self.aimodel.base_url, timeout=to, max_retries=0)
            else:
                self._client = OpenAI(api_key=self._config.api_key, timeout=to, max_retries=0)
            return True
        except Exception:
            self._client = None
            return False

    def count_tokens(self, msgs: list[dict[str, str]], *, output_format: type[BaseModel] | None = None) -> int:
        """メッセージ配列が消費するプロンプトトークンを概算する。"""
        if not msgs:
            return 0

        encoder = _get_token_encoder(self.aimodel.model)
        if encoder is None:
            # tiktoken が使えない環境では正確なトークン数は分からない
            return 0

        def token_len(text: str | None) -> int:
            if text is not None:
                try:
                    return len(encoder.encode(text))
                except Exception:
                    pass
            return 0

        tokens_per_message = 3
        tokens_per_name = 1
        if self.aimodel.model.startswith('gpt-3.5-turbo-0301'):
            tokens_per_message = 4
            tokens_per_name = -1

        total_tokens = 0
        for message in msgs:
            total_tokens += tokens_per_message
            total_tokens += token_len(message.get('content'))
            name = message.get('name')
            if name:
                total_tokens += tokens_per_name

        if output_format:
            try:
                fmt = json.dumps(output_format.model_json_schema()) # type: ignore[attr-defined]
                if fmt:
                    total_tokens += token_len(fmt) + tokens_per_message
            except Exception:
                pass

        total_tokens += 3  # assistant の初期トークン分
        return max(total_tokens, 0)

    def LLM(self, msg:list[dict[str,str]], *, output_format:type[BaseModel]|None=None) -> str:
        """LLMに問い合わせてJSON文字列を返す（OpenAI実装）。失敗時は例外。"""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")
        if self.get_input_strategy() == 'api':
            return self.LLM_responses_api(msg, output_format=output_format)
        return self.LLM_completions_api(msg, output_format=output_format)


    def convert_to_completion_input(self, msgs:List[Dict[str,str]]) -> List[ChatCompletionMessageParam]:
        xmsgs: List[ChatCompletionMessageParam] = []
        for m in msgs:
            role = m.get('role')
            content = m.get('content') or ""
            if role == ROLE_SYSTEM:
                xmsgs.append(ChatCompletionSystemMessageParam(role='system', content=content))
            elif role == ROLE_AI:
                xmsgs.append(ChatCompletionAssistantMessageParam(role='assistant', content=content))
            else:
                xmsgs.append(ChatCompletionUserMessageParam(role='user', content=content))
        return xmsgs


    def LLM_completions_api(self, msg:list[dict[str,str]], *, output_format:type[BaseModel]|None=None) -> str:
        """LLMに問い合わせてJSON文字列を返す（OpenAI実装）。失敗時は例外。"""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")

        xmsgs: List[ChatCompletionMessageParam] = self.convert_to_completion_input(msg)
        ofmt = Omit()
        if output_format:
            if self.aimodel.output_format == 'json_schema':
               ofmt = to_response_format(output_format, output_strict=True)
            else:
                try:
                    if hasattr(output_format, "to_json_format"):
                        fmt = output_format.to_json_format() # type: ignore
                        content = f"You must respond in JSON format exactly as specified: {fmt}." if fmt else ""
                    else:
                        fmt = output_format.model_json_schema() # type: ignore
                        content = f"You must respond in JSON schema exactly as specified: {fmt}." if fmt else ""
                except Exception as ex:
                    print(f"Failed to generate JSON schema: {ex}")
                    
                if content:
                    xmsgs.append(ChatCompletionSystemMessageParam(role='system', content=content))
                else:
                    raise RuntimeError(f"invalid object for output_format {type(output_format)}")
                output_format = None
                if self.aimodel.output_format=='json_object':
                    ofmt=ResponseFormatJSONObject(type='json_object')

        # モデル名が ^o[0-9] で始まる場合は temperature を 1.0 にする
        max_tokens = self.get_max_output_tokens() or Omit()
        if is_resoning_model(self.aimodel.model):
            temperature = None
        else:
            temperature = self._config.temperature or 0.2
        kwargs = {}
        if self.aimodel.reasoning:
            kwargs['reasoning_effort'] = self.aimodel.reasoning

        max_try:int = 8
        for ntry in range(max_try):
            try:
                if output_format is not None and ofmt is None:
                    resp = self._client.chat.completions.parse(
                        model=self.aimodel.model,
                        messages=xmsgs,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        response_format=output_format,
                        **kwargs
                    )
                else:
                    client: OpenAI = self._client  # type: ignore
                    resp = client.chat.completions.create(
                        model=self.aimodel.model,
                        messages=xmsgs,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        response_format=ofmt,
                        **kwargs
                    )
                break
            except Exception as ex:
                exmsg = str(ex)[:200]
                if isinstance(ex, (APIConnectionError, APITimeoutError)):
                    if ntry < max_try - 1:
                        print(f"OpenAI API connection error, retrying... ({ntry+1}/{max_try}): {exmsg}")
                        time.sleep(7.0*ntry)
                        continue
                if isinstance(ex, RateLimitError) and 'Daily quota exceeded' in str(ex):
                    raise LLMRateLimitError(f"OpenAI API rate limit exceeded: {exmsg}", retry_after=None, ) from ex
                raise LLMError(f"OpenAI API error: {exmsg}", ex)
        choice = resp.choices[0] if resp and len(resp.choices)>0 else None
        if choice is None:
            raise RuntimeError("empty response from model")
        if choice.finish_reason == "length" or choice.finish_reason == "content_filter":
            raise LLMError(f"OpenAI API error: finish_reason={choice.finish_reason}")
        content = choice.message.content if resp and resp.choices else None
        try:
            # CompletionUsage(completion_tokens=906, prompt_tokens=312, total_tokens=1218, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=896, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))
            u = getattr(resp, "usage", None)
            input_tokens = int(getattr(u, "prompt_tokens", 0)) if u is not None else 0
            output_tokens = int(getattr(u, "completion_tokens", 0)) if u is not None else 0
            detail1 = getattr(u, "completion_tokens_details", None)
            reasoning_tokens = int(getattr(detail1, "reasoning_tokens", 0)) if detail1 is not None else 0
            detail2 = getattr(u, "prompt_tokens_details", None)
            cache_read_tokens = int(getattr(detail2, "cached_tokens", 0)) if detail2 is not None else 0
            usage = LLMTokenUsage(
                prompt=input_tokens or 0,
                completion=output_tokens or 0,
                reasoning=reasoning_tokens or 0,
                cache_read=cache_read_tokens or 0,
            )
            self._set_last_usage_tokens(usage)
        except Exception:
            self._set_last_usage_tokens(None)
        if not content:
            raise RuntimeError("empty response from model")
        return content


    def split_mesgs(self, msgs:List[Dict[str,str]]) -> tuple[str|None, str|None, str|None]:
        sys_prompt = None
        prev_ai = None
        user_msg = None
        for m in msgs:
            role = m.get('role')
            content = m.get('content') or ""
            if role == ROLE_SYSTEM:
                if sys_prompt is None:
                    sys_prompt = content
            elif role == ROLE_AI:
                prev_ai = content
            else:
                user_msg = content
        return sys_prompt, prev_ai, user_msg


    def convert_to_response_input(self, msgs:List[Dict[str,str]]) -> tuple[str|None, ResponseInputParam]:
        xmsgs: ResponseInputParam = []
        sys_prompt = None
        for m in msgs:
            role = m.get('role')
            content = m.get('content') or ""
            if role == ROLE_SYSTEM:
                if sys_prompt is None:
                    sys_prompt = content
                else:
                    xmsgs.append(EasyInputMessageParam(role='system', content=content))
            elif role == ROLE_AI:
                xmsgs.append(EasyInputMessageParam(role='assistant', content=content))
            else:
                xmsgs.append(EasyInputMessageParam(role='user', content=content))
        return sys_prompt, xmsgs


    def LLM_responses_api(self, msg:list[dict[str,str]], *, output_format:type[BaseModel]|None=None) -> str:
        """LLMに問い合わせてJSON文字列を返す（OpenAI実装）。失敗時は例外。"""
        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")

        sys_prompt, prev_ai, user_msg = self.split_mesgs(msg)

        if self._prev_prompt==sys_prompt and self._prev_ai_msg==prev_ai and self._prev_response_id:
            prev_response_id = self._prev_response_id
            next_input = user_msg
        else:
            prev_response_id = None
            sys_prompt, next_input = self.convert_to_response_input(msg)

        if self.aimodel.output_format == 'json':
            if output_format:
                try:
                    fmt = output_format.to_json_format() # type: ignore
                    if fmt:
                        fmt = f"\n\nYou must respond in JSON format exactly as specified: {fmt}."
                        if isinstance(next_input, str):
                            next_input += fmt
                            output_format = None
                        elif isinstance(next_input, list) and len(next_input)>0:
                            last = next_input[-1]
                            if isinstance(last, dict):
                                last_content = str(last.get('content', "")) + fmt
                                last['content'] = last_content # type: ignore
                                output_format = None
                except Exception:
                    pass

        if is_resoning_model(self.aimodel.model):
            temperature = None
        else:
            temperature = self._config.temperature or 0.2

        kwargs = {}
        if self.aimodel.reasoning:
            kwargs['reasoning'] = {'effort': self.aimodel.reasoning}

        try:
            if not output_format:
                resp = self._client.responses.create(
                    model=self.aimodel.model,
                    temperature=temperature,
                    previous_response_id=prev_response_id,
                    instructions=sys_prompt,
                    input=next_input or "",
                    **kwargs
                )
            else:
                resp = self._client.responses.parse(
                    model=self.aimodel.model,
                    temperature=temperature,
                    previous_response_id=prev_response_id,
                    instructions=sys_prompt,
                    input=next_input or "",
                    text_format=output_format,
                    **kwargs
                )
        except Exception as ex:
            raise LLMError(f"OpenAI API error: {ex}", ex)
        # choice = resp.choices[0] if resp and len(resp.choices)>0 else None
        # if choice is None:
        #     raise RuntimeError("empty response from model")
        # if choice.finish_reason == "length" or choice.finish_reason == "content_filter":
        #     raise LLMError(f"OpenAI API error: finish_reason={choice.finish_reason}")
        content = resp.output_text
        # 次回のキャッシュ用
        if sys_prompt and resp and resp.id:
            self._prev_prompt = sys_prompt
            self._prev_ai_msg = content
            self._prev_response_id = resp.id
        try:
            # CompletionUsage(completion_tokens=906, prompt_tokens=312, total_tokens=1218, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=0, audio_tokens=0, reasoning_tokens=896, rejected_prediction_tokens=0), prompt_tokens_details=PromptTokensDetails(audio_tokens=0, cached_tokens=0))
            # ResponseUsage(input_tokens=3844, input_tokens_details=InputTokensDetails(cached_tokens=0), output_tokens=452, output_tokens_details=OutputTokensDetails(reasoning_tokens=0), total_tokens=4296)
            u = getattr(resp, "usage", None)
            input_tokens = int(getattr(u, "input_tokens", 0)) if u is not None else 0
            output_tokens = int(getattr(u, "output_tokens", 0)) if u is not None else 0
            detail1 = getattr(u, "output_tokens_details", None)
            reasoning_tokens = int(getattr(detail1, "reasoning_tokens", 0)) if detail1 is not None else 0
            detail2 = getattr(u, "input_tokens_details", None)
            cache_read_tokens = int(getattr(detail2, "cached_tokens", 0)) if detail2 is not None else 0
            usage = LLMTokenUsage(
                prompt=input_tokens or 0,
                completion=output_tokens or 0,
                reasoning=reasoning_tokens or 0,
                cache_read=cache_read_tokens or 0,
            )
            self._set_last_usage_tokens(usage)
        except Exception:
            self._set_last_usage_tokens(None)
        if not content:
            raise RuntimeError("empty response from model")
        return content

    def _diag_model_name(self) -> str:  # type: ignore[override]
        return self.aimodel.model
