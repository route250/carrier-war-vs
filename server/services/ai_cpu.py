"""
CPU向けAI実装（PvPエンジン上で動かす最小版）

目的:
- 既存の PvE 用 AI ルーチン（server/services/ai.py の plan_orders）を流用し、
  AIThreadABC 上で B 側ボットとして動作させる。

方針:
- 初回のみ `MatchStore.snapshot()` を用いて地形 `map` を取得・保持（以後は使い回し）。
- `build_state_payload(viewer_side=B)` で渡される `payload` から自軍の状態を再構築し、
  `plan_orders` へ必要情報を直接渡して命令を決定する。
- `plan_orders` は `PlayerOrders`（carrier_target / launch_target）を直接返却し提出する。

注意:
- 本ファイルは UI やルータへの変更を行わない。既存のフローに影響せずに差し込める。
"""

from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import List, Optional
import random


from server.services.ai_base import AIThreadABC
from server.schemas import PlayerOrders, AIModel, AIProvider
from server.schemas import Config, MatchStatePayload, Position, UnitState, CarrierState, SquadronState

from server.schemas import (
    CARRIER_MAX_HP,
    CARRIER_SPEED,
    CARRIER_HANGAR,
    CARRIER_RANGE,
    VISION_CARRIER,
    SQUAD_MAX_HP,
    SQUAD_SPEED,
    SQUADRON_RANGE,
    VISION_SQUADRON,
)

@dataclass
class EnemyAIState:
    patrol_ix: int = 0
    last_patrol_turn: int = 0

@dataclass
class IntelMarker:
    seen: bool
    pos: Position
    ttl: int

    @property
    def x(self) -> Optional[int]:
        try:
            return self.pos.x if (self.pos and self.pos.x >= 0 and self.pos.y >= 0) else None
        except Exception:
            return None

    @property
    def y(self) -> Optional[int]:
        try:
            return self.pos.y if (self.pos and self.pos.x >= 0 and self.pos.y >= 0) else None
        except Exception:
            return None

@dataclass
class EnemyMemory:
    carrier_last_seen: Optional[IntelMarker] = None
    enemy_ai: Optional[EnemyAIState] = None

@dataclass
class SquadronLight:
    id: str
    pos: Position

@dataclass
class PlayerObservation:
    visible_squadrons: List[SquadronLight] = field(default_factory=list)

@dataclass
class PlayerState:
    carrier: CarrierState
    squadrons: List[SquadronState] = field(default_factory=list)

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _pos_clamp( pos:Position, width:int, height:int) -> Position:
    return Position(x=_clamp(pos.x, 0, width - 1), y=_clamp(pos.y, 0, height - 1))

def _is_sea(grid: List[List[int]], pos:Position) -> bool:
    try:
        return grid[pos.y][pos.x] == 0
    except Exception:
        return False

def _chebyshev(a: Position, b: Position) -> int:
    return max(abs(a.x - b.x), abs(a.y - b.y))


def _offset_neighbors_odd_r(pos: Position) -> List[Position]:
    odd = pos.y & 1
    if odd:
        deltas = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (0, +1), (+1, +1)]
    else:
        deltas = [(+1, 0), (0, -1), (-1, -1), (-1, 0), (-1, +1), (0, +1)]
    return [Position(x=pos.x + dx, y=pos.y + dy) for dx, dy in deltas]


def _nearest_sea(grid: List[List[int]], pos: Position, w: int, h: int) -> Position:
    pos = _pos_clamp(pos, w, h)
    if _is_sea(grid, pos):
        return pos
    for r in range(1, 7):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                np = _pos_clamp( Position(x=pos.x+dx,y=pos.y+dy), w, h)
                if _is_sea(grid, np):
                    return np
    return pos

MODELS = AIProvider(name="CPU", models=[
            AIModel(name="Easy", model="easy", max_input_tokens=1, max_output_tokens=1),
            AIModel(name="Normal", model="normal", max_input_tokens=1, max_output_tokens=1),
            AIModel(name="Hard", model="hard", max_input_tokens=1, max_output_tokens=1),
    ])

class CarrierBotMedium(AIThreadABC):
    """PvP用CPUボット（最小実装）

    - AIThreadABC の `think(payload: dict)` を実装し、既存 `plan_orders` を呼び出す。
    - 地形 `map` は最初の呼び出し時に `store.snapshot()` で取得してキャッシュする。
    """

    def __init__(self, store, match_id: str, *, name: str = "CPU(Medium)", config: Config | None = None):
        super().__init__(store=store, match_id=match_id)
        self._memory: Optional[EnemyMemory] = None
        config = config if config is not None else Config(provider=MODELS.name)
        config.provider = MODELS.name
        config.llm_model = MODELS.to_model_or_default(config.llm_model)
        self.aimodel = MODELS.get_model(name=name) or MODELS.default()
        self._config: Config = config
        self._rand_seed: Optional[int] = None
        self._logs: list[list[str]] = []

    @property
    def name(self) -> str:
        ret = "CPU"
        if self._config and self._config.llm_model:
            ret += f"{self._config.llm_model}"
        return ret

    @staticmethod
    def get_model_names() -> AIProvider:
        return MODELS

    def _ensure_client_ready(self) -> bool:
        return True

    def startup(self) -> bool:
        return True

    def abort(self, payload:MatchStatePayload|None):
        pass

    def over(self, payload:MatchStatePayload|None):
        pass

    def think( self,  payload: MatchStatePayload ) -> PlayerOrders:
        logs: List[str] = []
        self._logs.append(logs)
        turn = payload.turn
        t0 = time.perf_counter()

        if not self.maparray:
            # マップが無ければ安全策として何も出さない
            return PlayerOrders()

        # 2) state payload から自軍（AI側）状態を復元
        ec, sq_list = self._payload_to_player_state(payload)
        if ec is None or sq_list is None:
            # 復元できない場合はノーオーダー
            return PlayerOrders()
        
        enemy_memory = self._memory
        rng = random.Random(self._rand_seed if self._rand_seed is not None else (turn * 7919))
        width = len(self.maparray[0]) if self.maparray and self.maparray[0] else 30
        height = len(self.maparray) if self.maparray else 30

        start = _pos_clamp(ec.pos, width, height)
        here = Position(x=start.x, y=start.y)

        # Known last seen player carrier position (if any)
        last_seen: Optional[Position] = None
        mem_in = enemy_memory.carrier_last_seen if enemy_memory else None
        if mem_in and mem_in.seen and mem_in.ttl > 0 and mem_in.pos is not None:
            last_seen = _pos_clamp( mem_in.pos, width, height)

        # Occupancy sets (known to server)
        enemy_occ = set([s.pos for s in sq_list if s.state not in ('onboard', "lost")])
        player_vis_occ = set()

        # 3) PlayerObservation（任意）: 可視編隊のみ最小反映（なければ None でOK）
        player_observation = self._payload_to_player_observation(payload)
        if player_observation:
            player_vis_occ = set([p.pos for p in player_observation.visible_squadrons])

        def cell_free(pos:Position) -> bool:
            return pos.in_bounds(width, height) and _is_sea(self.maparray, pos) and pos not in enemy_occ and pos not in player_vis_occ

        # Enemy carrier movement: 0..speed steps, biased away from last_seen if known
        steps = rng.randint(0, max(0, ec.speed))
        moved = False
        for _ in range(steps):
            nbs = [nxny for nxny in _offset_neighbors_odd_r(here) if cell_free(nxny)]
            if not nbs:
                break
            if last_seen is not None:
                curd = _chebyshev(here, last_seen)
                # score: prefer larger distance from last_seen, add tiny jitter
                scored = []
                for npos in nbs:
                    d = max(abs(npos.x - last_seen.x), abs(npos.y - last_seen.y))
                    scored.append(((d - curd) + rng.random() * 0.05, npos))
                scored.sort(key=lambda t: t[0], reverse=True)
                best_score, npos = scored[0]
                # If nothing improves distance, sometimes keep position (50%)
                if best_score <= 0 and rng.random() < 0.5:
                    break
            else:
                npos = rng.choice(nbs)
            here = Position(x=npos.x, y=npos.y)
            moved = True

        carrier_target: Optional[Position] = None
        if moved and (here.x != start.x or here.y != start.y):
            carrier_target = Position(x=here.x, y=here.y)
            logs.append("敵空母は回避運動")
            if last_seen is not None:
                logs[-1] = "敵空母は観測座標から離隔"

        # Squadron orders (mirror browser logic):
        # If have known target -> launch one base squadron. Else patrol every 3 turns using patrol points.
        launch_target: Optional[Position] = None
        active_cnt = sum(1 for s in sq_list if s.state not in ('onboard', "lost"))
        base_avail = next((s for s in sq_list if s.state == 'onboard' and (s.hp is None or s.hp > 0)), None)
        launched_to_known = False
        launched_patrol = False
        if active_cnt < ec.hangar and base_avail is not None:
            if last_seen is not None and mem_in and mem_in.ttl > 0:
                launch_target = Position(x=last_seen.x, y=last_seen.y)
                logs.append("敵編隊が出撃した気配")
                launched_to_known = True
            else:
                # Patrol cadence
                ai_in = enemy_memory.enemy_ai if enemy_memory and enemy_memory.enemy_ai else EnemyAIState()
                turns_since = turn - (ai_in.last_patrol_turn or 0)
                # patrol cadence depends on difficulty
                diff = (self._config.llm_model if self._config and self._config.llm_model else 'normal')
                cadence = 3 if diff == 'normal' else (2 if diff == 'hard' else 4)
                if turns_since >= cadence:
                    # Patrol waypoints: four corners + center
                    pts = [
                        Position(x=4, y=4),
                        Position(x=width - 5, y=4),
                        Position(x=4, y=height - 5),
                        Position(x=width - 5, y=height - 5),
                        Position(x=width // 2, y=height // 2),
                    ]
                    wp = pts[ai_in.patrol_ix % len(pts)]
                    tgt = _nearest_sea(self.maparray, wp, width, height)
                    launch_target = Position(x=tgt.x, y=tgt.y)
                    logs.append("敵編隊が索敵に出撃した気配")
                    launched_patrol = True

        # Memory evolution
        mem_out = EnemyMemory()
        # Carrier sighting TTL: if visible-now keep TTL (set by client). Otherwise decay by 1. If launched on known, decay once more.
        if mem_in:
            visible_now = mem_in.ttl >= 3  # client sets to 3 when visible
            ttl_next = mem_in.ttl if visible_now else max(0, mem_in.ttl - 1)
            if launched_to_known:
                ttl_next = max(0, ttl_next - 1)
            mem_out.carrier_last_seen = IntelMarker(seen=ttl_next > 0, pos=mem_in.pos, ttl=ttl_next)

        # Enemy AI patrol memory
        ai_in = enemy_memory.enemy_ai if enemy_memory and enemy_memory.enemy_ai else EnemyAIState()
        ai_out = EnemyAIState(patrol_ix=ai_in.patrol_ix, last_patrol_turn=ai_in.last_patrol_turn)
        if launched_patrol:
            ai_out.patrol_ix = ai_in.patrol_ix + 1
            ai_out.last_patrol_turn = turn
        mem_out.enemy_ai = ai_out
        # メモリ更新
        self._memory = mem_out or self._memory
        orders = PlayerOrders(carrier_target=carrier_target, launch_target=launch_target)
        return orders

    def cleanup(self) -> None:
        pass

    # --- helpers ---
    def _payload_to_player_state(self, payload: MatchStatePayload) -> tuple[CarrierState|None, list[SquadronState]|None]:
        try:
            units = payload.units
            carr = units.carrier
            if not carr:
                return None, None
            cx = carr.x
            cy = carr.y
            if cx is None or cy is None:
                return None, None
            carrier = CarrierState(
                id=carr.id or "C",
                side=self.side or "B",
                pos=Position(x=int(cx), y=int(cy)),
                hp=int(carr.hp) if carr.hp is not None else CARRIER_MAX_HP,
                max_hp=int(carr.max_hp) if carr.max_hp is not None else CARRIER_MAX_HP,
                speed=int(carr.speed) if carr.speed is not None else CARRIER_SPEED,
                fuel=int(carr.fuel) if carr.fuel is not None else CARRIER_RANGE,
                vision=int(carr.vision) if carr.vision is not None else VISION_CARRIER,
            )

            sq_list = []
            for sq in units.squadrons or []:
                pos_x = sq.x
                pos_y = sq.y
                squad = SquadronState(
                    id=sq.id or "SQ",
                    side=self.side or "B",
                    hp=int(sq.hp) if sq.hp is not None else SQUAD_MAX_HP,
                    max_hp=int(sq.max_hp) if sq.max_hp is not None else SQUAD_MAX_HP,
                    speed=int(sq.speed) if sq.speed is not None else SQUAD_SPEED,
                    fuel=int(sq.fuel) if sq.fuel is not None else SQUADRON_RANGE,
                    vision=int(sq.vision) if sq.vision is not None else VISION_SQUADRON,
                    state=sq.state or 'onboard',
                )
                if pos_x is not None and pos_y is not None:
                    squad.pos = Position(x=int(pos_x), y=int(pos_y))
                sq_list.append(squad)

            return carrier, sq_list
        except Exception:
            return None, None

    def _payload_to_player_observation(self, payload: MatchStatePayload|None) -> Optional[PlayerObservation]:
        try:
            # 現状の state には敵編隊の最小情報を返す設計（intel）だが、
            # ここでは安全側へ倒して None または空観測を返す。
            # 将来、`intel.squadrons` 等が付与されたら変換を実装。
            return None
        except Exception:
            return None

    def to_usage_dict(self) -> dict[str,float]:
        return {}

    def save_history(self, logspath: str):
        pass