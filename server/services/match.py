import time
import traceback
import uuid
import asyncio
import threading
import traceback
import json
import os
import sys
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Optional

from server.schemas import (
    Config,
    MatchStatePayload,
    MatchCreateRequest,
    MatchCreateResponse,
    MatchJoinRequest,
    MatchJoinResponse,
    MatchListItem,
    MatchListResponse,
    MatchMode,
    MatchOrdersRequest,
    MatchOrdersResponse,
    MatchStateResponse,
    MatchStatus,
    UnitState,
    CarrierState,
    SquadronState,
    Position,
    PlayerOrders,
    AIProvider,
)

from server.services.ai_base import AIThreadABC
from server.services.ai_cpu import CarrierBotMedium
from server.services.ai_llm_base import LLMBase
from server.services.ai_openai import CarrierBotOpenAI, OpenAIConfig
from server.services.ai_anthropic import CarrierBotAnthropic, AnthropicConfig
from server.services.ai_gemini import CarrierBotGemini, GeminiConfig
from server.services.turn import GameBord, IntelReport

# Debug flag: enable when running tests or when env var CARRIER_WAR_DEBUG is set
DEBUG = bool(os.getenv('CARRIER_WAR_DEBUG')) or ('unittest' in sys.modules) or ('PYTEST_CURRENT_TEST' in os.environ)

def _dbg(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


@dataclass
class ActivityLog:
    last_turn: int = 0
    json_errors: int = 0
    carrier_request_count: int = 0
    carrier_request_errors: int = 0
    squadron_request_count: int = 0
    squadron_request_errors: int = 0
    model_name: str|None = None
    spotted_enemy_turn: int | None = None
    trip_pos: dict[str,Position] = field(default_factory=dict)
    trip_count: dict[str,dict[int,int]] = field(default_factory=dict)

    def add_trip(self, turn, unit_id:str, pos:Position) -> None:
        prev = self.trip_pos.get(unit_id)
        if prev is not None and prev != pos:
            a = self.trip_count.setdefault(unit_id,{})
            a[turn] = prev.hex_distance(pos)
        self.trip_pos[unit_id] = pos

    def set_spotted_enemy(self, turn:int) -> None:
        if self.spotted_enemy_turn is None:
            self.spotted_enemy_turn = turn

    def add_errors(self, c:bool, sq:bool ) -> None:
        if c:
            self.carrier_request_errors += 1
        if sq:
            self.squadron_request_errors += 1

    def add_orders(self, orders: PlayerOrders|None) -> None:
        if orders is not None:
            if orders.carrier_target is not None:
                self.carrier_request_count += 1
            if orders.launch_target is not None:
                self.squadron_request_count += 1

    def add_activity(self, payload: MatchStatePayload) -> None:
        if payload.you_name:
            if self.model_name is None:
                self.model_name = payload.you_name
            elif self.model_name != payload.you_name:
                raise ValueError("name changed")
        if payload.turn == self.last_turn:
            return
        elif payload.turn == self.last_turn + 1:
            self.last_turn = payload.turn
            if payload.units.carrier and payload.units.carrier.x is not None and payload.units.carrier.y is not None:
                self.add_trip(payload.turn, payload.units.carrier.id, Position(x=payload.units.carrier.x, y=payload.units.carrier.y))
            for sq in (payload.units.squadrons or []):
                if sq.x is not None and sq.y is not None:
                    self.add_trip(payload.turn, sq.id, Position(x=sq.x, y=sq.y))
            if payload.intel.carrier and payload.intel.carrier.x is not None and payload.intel.carrier.y is not None:
                self.set_spotted_enemy(payload.turn)
        else:
            raise ValueError("turn skipped")

    def get_trip_counts(self, unit_id:str ):
        data = [ (turn,trip) for turn,trip in self.trip_count.get(unit_id,{}).items() ]
        return sorted(data)

    def to_dict(self):
        return {
            "last_turn": self.last_turn,
            "json_errors": self.json_errors,
            "carrier_request_count": self.carrier_request_count,
            "carrier_request_errors": self.carrier_request_errors,
            "squadron_request_count": self.squadron_request_count,
            "squadron_request_errors": self.squadron_request_errors,
            "spotted_enemy_turn": self.spotted_enemy_turn,
            "trip_count": { uid:self.get_trip_counts(uid) for uid in self.trip_count.keys() }
        }

@dataclass
class PlayerSlot:
    token: Optional[str] = None
    name: Optional[str] = None
    orders: PlayerOrders | None = None  # raw dict from PlayerOrders for now


@dataclass
class Match:
    match_id: str
    mode: MatchMode
    map: GameBord
    status: MatchStatus = "waiting"
    config: Optional[Config] = None
    created_at: int = field(default_factory=lambda: int(time.time()))
    side_a: PlayerSlot = field(default_factory=PlayerSlot)
    side_b: PlayerSlot = field(default_factory=PlayerSlot)
    activity_a: ActivityLog = field(default_factory=ActivityLog)
    activity_b: ActivityLog = field(default_factory=ActivityLog)
    ai_threads: list[Optional[AIThreadABC]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock, repr=False)
    subscribers_map: dict[asyncio.Queue[str], Optional[str]] = field(default_factory=dict, repr=False)
    last_report: Optional[dict[str, IntelReport]] = None

    def close(self):
        # Stop AI threads
        for thread in self.ai_threads:
            try:
                if thread and thread.is_alive():
                    thread.stop()
            except Exception:
                pass
        self.ai_threads = []

    def leave(self, token: str ) -> bool:
        changed = False
        for thread in self.ai_threads:
            try:
                if thread and thread.token == token:
                    thread.stop()
            except Exception:
                pass
        if self.side_a.token == token:
            self.side_a = PlayerSlot()
            changed = True
        if self.side_b.token == token:
            self.side_b = PlayerSlot()
            changed = True
        if not changed:
            return False
        self.status = "over"
        return True

    def has_open_slot(self) -> bool:
        return not self.side_a.token or not self.side_b.token

    def side_for_token(self, token: str) -> Optional[str]:
        if self.side_a.token == token:
            return "A"
        if self.side_b.token == token:
            return "B"
        return None

    def add_json_errors(self, token: str ) -> None:
        side = self.side_for_token(token)
        if side == 'A':
            self.activity_a.json_errors += 1
        elif side=='B':
            self.activity_b.json_errors += 1

    def set_orders(self, token: str, orders: PlayerOrders|None, *, dry_run: bool = False) -> list[str]:

        if self.status != "active":
            return ["match not active"] # TODO: aiを止める必要あり
        side = self.side_for_token(token)
        if side != "A" and side != "B":
            return ["invalid token"]

        # check carrier target validity
        msg1,msg2 = self.map.validate_orders(side, orders)
        if side == "A":
            self.activity_a.add_errors( len(msg1)>0, len(msg2)>0 )
        if side == "B":
            self.activity_b.add_errors( len(msg1)>0, len(msg2)>0 )
        msg = msg1 + msg2
        if msg:
            return msg
        if side == "A":
            if not dry_run:
                self.activity_a.add_orders(orders)
                self.side_a.orders = orders
        elif side == "B":
            if not dry_run:
                self.activity_b.add_orders(orders)
                self.side_b.orders = orders
        else:
            return ["invalid token"]
        return []

    def get_activity_dict(self, token: str|None ) -> dict:
        side = self.side_for_token(token) if token else None
        if side == 'A' or token == 'A':
            return self.activity_a.to_dict()
        elif side == 'B' or token == 'B':
            return self.activity_b.to_dict()
        return {}

    def _resolve_turn_minimal(self) -> None:
        # Move carriers towards targets if provided
        orders = [self.side_a.orders or PlayerOrders(), self.side_b.orders or PlayerOrders()]
        self.last_report = self.map.turn_forward(orders)  # use existing turn logic to apply orders
        # Check game over condition (any carrier destroyed)
        if self.map.is_over():
            self.status = "over"

    def get_state(self, token:str|None ) -> MatchStatePayload:
        side = self.side_for_token(token) if token else None
        # API は従来通り dict を返す
        return self.build_state_payload(viewer_side=side)

    def build_state_payload(self, viewer_side: Optional[str] = None) -> MatchStatePayload:
        waiting_for = "none"
        game_result = None
        if self.status == "active":
            a_has = self.side_a.orders is not None
            b_has = self.side_b.orders is not None
            if not a_has and not b_has:
                waiting_for = "orders"
            elif not a_has or not b_has:
                if viewer_side == "A" and not a_has or viewer_side == "B" and not b_has:
                    waiting_for = "you"
                else:
                    waiting_for = "opponent"
        if self.status == "over":
            if self.map.get_result() == viewer_side:
                game_result = "win"
            elif self.map.get_result() is None:
                game_result = "draw"
            else:
                game_result = "lose"
        aw = self.map.W
        ah = self.map.H
        my_units, other_units = self.map.to_payload(viewer_side)
        # 一旦モデルを作成
        payload = MatchStatePayload(
            match_id=self.match_id,
            status=self.status,
            turn=self.map.turn,
            max_turn=self.map.max_turn,
            waiting_for=waiting_for,
            result=game_result,
            map_w=aw,
            map_h=ah,
            units=my_units,
            intel=other_units,
        )
        # 補助情報（視点別の表示用名）
        try:
            if viewer_side in ("A", "B"):
                if viewer_side == "A":
                    payload.you_name = self.side_a.name or None
                    # 相手名は、B側の名前。未設定時は PvE の場合にプロバイダ名をフォールバック
                    payload.opponent_name = self.side_b.name or None
                    if not payload.opponent_name:
                        try:
                            payload.opponent_name = self.config.provider if self.config else None
                        except Exception:
                            pass
                elif viewer_side == "B":
                    payload.you_name = self.side_b.name or None
                    payload.opponent_name = self.side_a.name or None
                    if not payload.opponent_name:
                        try:
                            payload.opponent_name = self.config.provider if self.config else None
                        except Exception:
                            pass
        except Exception:
            pass
        # Attach per-viewer logs (previous turn) if available
        try:
            if viewer_side in ("A", "B") and self.last_report is not None:
                rep = self.last_report.get(viewer_side)
                if rep:
                    # クライアント側で重複追加を避けるため、常に最新ターンのstateに含めるだけにする
                    payload.logs = list(rep.logs)
        except Exception:
            pass

        # PvE かつ 人間視点にAI診断を付与（注文JSONは含めない）
        try:
            if self.mode == "pve" and viewer_side in ("A", "B"):
                # AIスレッドのサイドが viewer と逆側のものを探す
                ai_diag = None
                for th in (self.ai_threads or []):
                    if not isinstance(th, LLMBase):
                        continue
                    try:
                        th_side = th.side
                        if th_side and th_side != viewer_side:
                            ai_diag = th.get_ai_diag()
                            break
                    except Exception:
                        pass
                if ai_diag:
                    payload.ai_diag = ai_diag  # type: ignore[assignment]
        except Exception:
            pass

        if viewer_side == 'A':
            self.activity_a.add_activity(payload)
        elif viewer_side == 'B':
            self.activity_b.add_activity(payload)
        return payload


    def _broadcast_state(self) -> None:
        try:
            if len(self.subscribers_map) > 0:
                subs = dict(self.subscribers_map)
                for q,token in subs.items():
                    try:
                        side = self.side_for_token(token) if token else None
                        payload = self.build_state_payload(viewer_side=side)
                        print(f"broadcast to {side}: wait_for:{payload.waiting_for}")
                        data = json.dumps(payload.model_dump(), ensure_ascii=False)
                        q.put_nowait(data)
                    except Exception:
                        pass
            if self.ai_threads:
                for thread in self.ai_threads:
                    try:
                        if thread and thread.is_ready():
                            payload = self.build_state_payload(viewer_side=thread.side)
                            # print(f"notify to {thread.side}: wait_for:{payload.waiting_for}")
                            payload.units.turn_visible = None  # AIには視界情報は不要
                            thread.put_payload(payload)
                    finally:
                        pass

        except Exception:
            traceback.print_exc()
            pass

class MatchStore:
    def __init__(self) -> None:
        self._matches: Dict[str, Match] = {}
        self._lobby_subs: list[asyncio.Queue[str]] = []
        # 提供可能なAIのカタログ
        self._ai_catalog = [
            CarrierBotMedium.get_model_names(),
            CarrierBotOpenAI.get_model_names(),
            CarrierBotAnthropic.get_model_names(),
            CarrierBotGemini.get_model_names()
        ]

    def create(self, req: MatchCreateRequest) -> MatchCreateResponse:
        from server.services.hexmap import HexArray, generate_connected_map as hex_generate_connected_map
        # Generate connected map and carve safe sea around spawn points
        mid = str(uuid.uuid4())
        W = 30; H = 30
        map = HexArray(W, H)
        hex_generate_connected_map(map, blobs=10)
        # Place carriers and ensure sea around them
        a_units = create_units("A", 3,3 )
        b_units = create_units("B", W-4, H-4 )
        bord = GameBord(map, [a_units, b_units], log_id=mid)
        m = Match(match_id=mid, mode=req.mode or "pvp", map=bord, config=req.config)
        # creator occupies side A by default
        token = ""
        side = ""
        if m.mode != 'eve':
            side='A'
            token = str(uuid.uuid4())
            m.side_a.token = token
            m.side_a.name = req.display_name
        self._matches[mid] = m

        # broadcast lobby list update
        self._broadcast_lobby_list()

        if req.mode == "pve" and req.config:
            try:
                # PvE時は指定AIを起動。未指定は従来のルールベース(normal)
                provider = (req.config.provider if req.config and req.config.provider else 'CPU').lower()
                if provider == 'openai':
                    cfg = OpenAIConfig(model=req.config.llm_model) # type: ignore[arg-type]
                    bot = CarrierBotOpenAI(store=self, match_id=mid, config=cfg)
                elif provider == 'anthropic':
                    cfg = AnthropicConfig(model=req.config.llm_model) # type: ignore[arg-type]
                    bot = CarrierBotAnthropic(store=self, match_id=mid, config=cfg)
                elif provider == 'gemini':
                    cfg = GeminiConfig(model=req.config.llm_model) # type: ignore[arg-type]
                    bot = CarrierBotGemini(store=self, match_id=mid, config=cfg)
                else:
                    # ルールベース。難易度は config.llm_model を参照（未指定は 'normal'）。
                    bot = CarrierBotMedium(store=self, match_id=mid, config=req.config)
                m.ai_threads.append(bot)
                # 常に専用スレッドでAIを起動（イベントループ有無に依存しない）
                t = threading.Thread(target=bot.run, daemon=True)
                t.start()
            except Exception:
                traceback.print_exc()

        return MatchCreateResponse(
            match_id=mid,
            player_token=token,
            side="A",
            status=m.status,
            mode=m.mode,
            config=req.config,
        )

    def get_match_list(self) -> MatchListResponse:
        items = []
        for m in self._matches.values():
            items.append(
                MatchListItem(
                    match_id=m.match_id,
                    status=m.status,
                    mode=m.mode,
                    has_open_slot=m.has_open_slot(),
                    created_at=m.created_at,
                    config=m.config,
                    players=[p for p in [m.side_a.name, m.side_b.name] if p],
                )
            )
        return MatchListResponse(matches=items)

    # 提供AIの一覧（プロバイダ単位）
    def get_ai_list(self):
        from server.schemas import AIListResponse

        def is_expensive(model) -> bool:
            price_in = model.input_price
            price_out = model.output_price
            if price_in is not None and price_in > 1.0:
                return True
            if price_out is not None and price_out > 10.0:
                return True
            return False

        filtered_providers: list[AIProvider] = []
        for provider in self._ai_catalog:
            allowed_models = [m for m in provider.models if not is_expensive(m)]
            if not allowed_models:
                continue

            default_idx = 0
            if provider.models and 0 <= provider.default_index < len(provider.models):
                default_model = provider.models[provider.default_index]
                for idx, model in enumerate(allowed_models):
                    if model.model == default_model.model:
                        default_idx = idx
                        break

            filtered_providers.append(
                AIProvider(name=provider.name, models=allowed_models, default_index=default_idx)
            )

        return AIListResponse(providers=filtered_providers)

    def join(self, match_id: str, req: MatchJoinRequest) -> MatchJoinResponse:
        m = self._matches[match_id]
        with m.lock:
            if not m.side_b.token:
                side = "B"
                token = str(uuid.uuid4())
                m.side_b.token = token
                m.side_b.name = req.display_name
            elif not m.side_a.token:
                side = "A"
                token = str(uuid.uuid4())
                m.side_a.token = token
                m.side_a.name = req.display_name
            else:
                # already full
                raise KeyError("match full")
            # if both present, activate
            if m.side_a.token and m.side_b.token:
                m.status = "active"
            # broadcast lobby list update
            self._broadcast_lobby_list()
            # broadcast updated match state so creator gets immediately notified
            m._broadcast_state()
            return MatchJoinResponse(match_id=m.match_id, player_token=token, side=side, status=m.status)

    def state(self, match_id: str, token: Optional[str] = None) -> MatchStateResponse:
        m = self._matches[match_id]
        payload = m.get_state(token)
        ret = MatchStateResponse( match_id=payload.match_id, turn=payload.turn, status=payload.status )
        return ret

    def is_match_waiting(self, match_id: str) -> bool:
        m = self._matches[match_id]
        return m is not None and m.status == "waiting"

    def is_match_active(self, match_id: str) -> bool:
        m = self._matches[match_id]
        return m is not None and m.status == "active"

    def subscribe(self, match_id: str, token: Optional[str]) -> asyncio.Queue[str]:
        """ start sse session """
        m = self._matches[match_id]
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        with m.lock:
            m.subscribers_map[q] = token
        return q

    def snapshot(self, match_id: str, token: Optional[str] = None) -> MatchStatePayload:
        """ first data for sse session"""
        m = self._matches[match_id]
        side = m.side_for_token(token) if token else None
        payload = m.build_state_payload(viewer_side=side)
        return payload

    def get_map_array(self, match_id: str) -> list[list[int]]:
        m = self._matches[match_id]
        return m.map.get_map_array() if m else []

    def unsubscribe(self, match_id: str, q: asyncio.Queue[str]) -> None:
        """ end of sse session """
        m = self._matches.get(match_id)
        if not m:
            return
        with m.lock:
            token = m.subscribers_map.pop(q, None)
            # If this subscriber was tied to a player token and no other
            # subscriptions remain for that token, consider that player left
            if token:
                remaining = [tok for tok in m.subscribers_map.values() if tok == token]
                if not remaining:
                    # clear player's slot
                    if m.side_a.token == token:
                        m.side_a = PlayerSlot()  # reset
                    elif m.side_b.token == token:
                        m.side_b = PlayerSlot()
                    # update status
                    if not (m.side_a.token and m.side_b.token):
                        if m.status != "over":
                            m.status = "waiting"
                    # if no players remain, delete match entirely
                    if not m.side_a.token and not m.side_b.token:
                        # delete and broadcast lobby list, then return
                        try:
                            del self._matches[m.match_id]
                        except Exception:
                            pass
                        #
                        m.close()

                        self._broadcast_lobby_list()
                        return
                    # broadcast lobby list and updated state to remaining subscribers
                    self._broadcast_lobby_list()

                    m._broadcast_state()

    # --- Lobby SSE ---
    def lobby_subscribe(self) -> asyncio.Queue[str]:
        """ start lobby sse session """
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
        self._lobby_subs.append(q)
        return q

    def lobby_unsubscribe(self, q: asyncio.Queue[str]) -> None:
        try:
            self._lobby_subs.remove(q)
        except ValueError:
            pass

    def _broadcast_lobby_list(self) -> None:
        try:
            payload = {"type": "list", "matches": self.get_match_list().model_dump().get("matches", [])}
            data = json.dumps(payload, ensure_ascii=False)
            subs = list(self._lobby_subs)
            for q in subs:
                try:
                    q.put_nowait(data)
                except Exception:
                    pass
        except Exception:
            pass

    def leave(self, match_id: str, token: str) -> None:
        m = self._matches.get(match_id)
        if not m:
            raise KeyError("match not found")
        with m.lock:
            changed = m.leave(token)
            if not changed:
                return
            # if no players remain, delete match
            if not m.side_a.token and not m.side_b.token:
                try:
                    del self._matches[m.match_id]
                except Exception:
                    pass
                m.close()
                self._broadcast_lobby_list()
                return
            # otherwise broadcast updates
            self._broadcast_lobby_list()

            m._broadcast_state()

    def validate_orders(self, match_id: str, req: MatchOrdersRequest) -> list[str]:
        m = self._matches[match_id]
        with m.lock:
            msgs = m.set_orders(req.player_token, req.player_orders, dry_run=True)
            return msgs

    def add_json_errors(self, match_id: str|None, token: str|None) -> None:
        m = self._matches.get(match_id) if match_id else None
        if m and token:
            m.add_json_errors(token)

    def submit_orders(self, match_id: str, req: MatchOrdersRequest) -> MatchOrdersResponse:
        m = self._matches[match_id]
        with m.lock:
            msgs:list[str] = m.set_orders(req.player_token, req.player_orders)
            if msgs:
                _dbg(f"Order validation failed: {msgs}")
                return MatchOrdersResponse(accepted=False, status=m.status, turn=m.map.turn, logs=msgs)

            # Resolve turn only when both sides submitted (ready)
            if m.status == "active" and (m.side_a.orders is not None and m.side_b.orders is not None):
                try:
                    m._resolve_turn_minimal()
                except Exception:
                    # even if resolution fails, advance to avoid deadlock
                    pass
                # clear orders for next turn
                m.side_a.orders = None
                m.side_b.orders = None
            m._broadcast_state()

        return MatchOrdersResponse(accepted=True, status=m.status, turn=m.map.turn)


store = MatchStore()

# ---------- Internal helpers ----------
def create_units(side:str, cx: int, cy: int) -> list[UnitState]:
    un:list[UnitState]=[]
    i=1
    carrier = CarrierState(id=f"{side}C{i}", side=side, pos=Position(x=cx, y=cy))
    un.append(carrier)
    for s in range(0, carrier.hangar):
        un.append(SquadronState(id=f"{side}SQ{s+1}", side=side))
    return un
