from dataclasses import dataclass
import math
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
try:
    # pydantic v2 provides computed_field for including derived values in serialization
    from pydantic import computed_field
except Exception:  # fallback for environments without pydantic v2
    computed_field = None  # type: ignore

INF:int = 10**8

CARRIER_MAX_HP = 100
CARRIER_SPEED = 2
CARRIER_HANGAR = 2
CARRIER_RANGE = 99999
VISION_CARRIER = 4

SQUAD_MAX_HP = 40
SQUAD_SPEED = 4
SQUADRON_RANGE = 22
VISION_SQUADRON = 5

class Position(BaseModel,frozen=True):

    x: int
    y: int

    def __le__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) <= (other.x, other.y)

    def __gt__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) > (other.x, other.y)

    def __ge__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) >= (other.x, other.y)
    def __lt__(self, other):
        if not isinstance(other, Position):
            return NotImplemented
        return (self.x, self.y) < (other.x, other.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Position):
            return self.x == other.x and self.y == other.y
        return False

    @staticmethod
    def invalid() -> 'Position':
        return Position(x=-1, y=-1)

    def is_valid(self) -> bool:
        return self.x>=0 and self.y>=0

    @staticmethod
    def new(p1:'int|tuple[int,int]|Position', p2:int|None=None) -> 'Position':
        if isinstance(p1, Position):
            return Position(x=p1.x, y=p1.y)
        elif isinstance(p1, (tuple, list)) and len(p1) == 2:
            return Position(x=p1[0], y=p1[1])
        elif isinstance(p1, int) and isinstance(p2, int):
            return Position(x=p1, y=p2)
        else:
            raise TypeError(f"invalid parameters to Position.new {p1}, {p2}")


    def in_bounds(self, w: int, h: int) -> bool:
        return 0 <= self.x < w and 0 <= self.y < h


    def hex_distance(self, p1:'int|tuple[int,int]|Position', p2:int|None=None) -> int:
        if isinstance(p1, Position):
            x,y = p1.x, p1.y
        elif isinstance(p1, (tuple, list)) and len(p1) == 2:
            x,y = p1
        elif isinstance(p1, int) and isinstance(p2, int):
            x,y = p1,p2
        else:
            raise TypeError("Other must be a Position")
        aq, ar = Position._offset_to_axial(self.x, self.y)
        bq, br = Position._offset_to_axial(x, y)
        ax, ay, az = Position._axial_to_cube(aq, ar)
        bx, by, bz = Position._axial_to_cube(bq, br)
        return Position._cube_distance(ax, ay, az, bx, by, bz)

    @staticmethod
    def _offset_to_axial(col: int, row: int):
        q = col - ((row - (row & 1)) >> 1)
        r = row
        return q, r

    @staticmethod
    def _axial_to_cube(q: int, r: int):
        x = q
        z = r
        y = -x - z
        return x, y, z

    @staticmethod
    def _cube_distance(ax: int, ay: int, az: int, bx: int, by: int, bz: int):
        return max(abs(ax - bx), abs(ay - by), abs(az - bz))

    @staticmethod
    def _hex_distance(pos1: 'Position', pos2: 'Position') -> int:
        aq, ar = Position._offset_to_axial(pos1.x, pos1.y)
        bq, br = Position._offset_to_axial(pos2.x, pos2.y)
        ax, ay, az = Position._axial_to_cube(aq, ar)
        bx, by, bz = Position._axial_to_cube(bq, br)
        return Position._cube_distance(ax, ay, az, bx, by, bz)

    def offset_neighbors(self):
        odd = self.y & 1
        if odd:
            deltas = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (0, +1), (+1, +1)]
        else:
            deltas = [(+1, 0), (0, -1), (-1, -1), (-1, 0), (-1, +1), (0, +1)]
        for dx, dy in deltas:
            yield Position(x=self.x + dx, y=self.y + dy)

    def angle_to(self, other: 'Position') -> float:
        """
        selfからotherへの角度（ラジアン）を返す。
        """
        x0, y0 = self.center_xy()
        x1, y1 = other.center_xy()
        dx = x1 - x0
        dy = y1 - y0
        return math.atan2(dy, dx)

    def center_xy(self) -> tuple[float,float]:
        """
        六角形の中心座標を返す。
        """
        x = self.x * 1.0
        y = self.y * math.sqrt(3)/2
        if self.y & 1:
            x += 0.5
        return (x, y)

    def vector(self, other: 'Position') -> tuple[float,float]:
        """
        selfからotherへのベクトルを返す。
        """
        x0, y0 = self.center_xy()
        x1, y1 = other.center_xy()
        return (x1 - x0, y1 - y0)

    def length(self, other: 'Position') -> float:
        """
        selfからotherへの距離を返す。
        """
        dx, dy = self.vector(other)
        return math.hypot(dx, dy)

    def normalized_vector(self, other: 'Position') -> tuple[float,float]:
        """
        selfからotherへの正規化ベクトルを返す。
        """
        dx, dy = self.vector(other)
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return (0.0, 0.0)
        return (dx / length, dy / length)

class UnitState(BaseModel):
    id: str
    side: str
    pos: Position
    hp: int
    max_hp: int
    speed: int
    fuel: int
    vision: int
    origin: Position|None = None
    _target: Position|None = None

    @property
    def target(self) -> Position|None:
        return self._target

    @target.setter
    def target(self, value: Position|None) -> None:
        if value is not None:
            if value != self._target:
                # 目標が変わる場合、現在位置を起点にセット
                if self.pos is None:
                    raise ValueError("Cannot set target when pos is None")
                self.origin = Position(x=self.pos.x, y=self.pos.y)
                self._target = value
        else:
            self._target = None
            self.origin = None

    def is_active(self) -> bool:
        return self.hp > 0 and self.pos is not None and self.pos.x >= 0 and self.pos.y >= 0

    def can_see_enemy(self, enemy:'UnitState') -> bool:
        """Return True if tile (x,y) is visible to the player (carrier or active squadrons).
        """
        return self.hex_distance(enemy) <= self.vision

    def is_visible_to_player(self, other:'UnitState') -> bool:
        """Return True if tile (x,y) is visible to the player (carrier or active squadrons).
        """
        return self.hex_distance(other) <= self.vision

    def hex_distance(self, other:'UnitState|Position') -> int:
        if self.is_active():
            if isinstance(other, Position) and other.x>=0 and other.y>=0:
                return self.pos.hex_distance(other)
            elif isinstance(other, UnitState) and other.is_active():
                return self.pos.hex_distance(other.pos)
        return INF

    # Flattened coordinates for client convenience (read-only, derived from pos)
    if computed_field:
        @computed_field  # type: ignore[misc]
        def x(self) -> Optional[int]:
            try:
                return self.pos.x if (self.pos and self.pos.x >= 0 and self.pos.y >= 0) else None
            except Exception:
                return None

        @computed_field  # type: ignore[misc]
        def y(self) -> Optional[int]:
            try:
                return self.pos.y if (self.pos and self.pos.x >= 0 and self.pos.y >= 0) else None
            except Exception:
                return None

class CarrierState(UnitState):
    hp: int = CARRIER_MAX_HP
    max_hp: int = CARRIER_MAX_HP
    speed: int = CARRIER_SPEED
    fuel: int = CARRIER_RANGE
    vision: int = VISION_CARRIER
    hangar: int = CARRIER_HANGAR


class SquadronState(UnitState):
    pos: Position = Position.invalid()
    hp: int = SQUAD_MAX_HP
    max_hp: int = SQUAD_MAX_HP
    speed: int = SQUAD_SPEED
    fuel: int = SQUADRON_RANGE
    vision: int = VISION_SQUADRON
    state: Literal['onboard', "outbound", "engaging", "returning", "lost"] = 'onboard'

    def is_active(self) -> bool:
        return super().is_active() and self.state != "lost" and self.state != 'onboard'

class Config(BaseModel):
    provider: str
    llm_model: str|None = None
    time_ms: Optional[int] = 50

class PlayerOrders(BaseModel):
    carrier_target: Optional[Position] = None
    launch_target: Optional[Position] = None


# === PvP Match (skeleton) ===
# まずは最低限の型を用意（段階的に拡張）
MatchMode = Literal["pve", "pvp", "eve"]
MatchStatus = Literal["waiting", "active", "over"]


class MatchCreateRequest(BaseModel):
    mode: Optional[MatchMode] = "pvp"
    config: Optional[Config] = None
    display_name: Optional[str] = None


class MatchCreateResponse(BaseModel):
    match_id: str
    player_token: str
    side: Literal["A", "B"] = "A"
    status: MatchStatus = "waiting"
    mode: MatchMode = "pvp"
    config: Optional[Config] = None


class MatchListItem(BaseModel):
    match_id: str
    status: MatchStatus
    mode: MatchMode
    has_open_slot: bool
    created_at: int
    config: Optional[Config] = None
    players: list[str] = []


class MatchListResponse(BaseModel):
    matches: List[MatchListItem] = []


class MatchJoinRequest(BaseModel):
    display_name: Optional[str] = None


class MatchJoinResponse(BaseModel):
    match_id: str
    player_token: str
    side: Literal["A", "B"]
    status: MatchStatus


# === AI Catalog ===

@dataclass
class LLMBaseConfig:
    """LLM共通の設定項目"""
    model: str|None = None
    api_key: str|None = None
    temperature: float|None = None
    max_input_tokens: int|None = None
    max_output_tokens: int|None = None
    input_strategy: Literal['truncate', 'summarize', 'api']|None = None  # 入力が長すぎる場合の挙動
    language: Literal['ja', 'en'] = 'ja'  # 使用する言語

class AIModel(BaseModel, frozen=True):
    name: str  # display label for UI
    model: str    # internal model ID
    max_input_tokens: int
    max_output_tokens: int| None = None  # None means unlimited
    temperature: float = 0.2
    input_strategy: Literal['truncate', 'summarize', 'api'] = 'api'  # 入力が長すぎる場合の挙動
    reasoning: Literal['minimal', 'low', 'medium', 'high'] | None = None  # 推論モデルか？
    input_price: float | None = None  # per 1M tokens
    cached_price: float | None = None # per 1M tokens
    cache_write_price: float | None = None # per 1M tokens
    output_price: float | None = None # per 1M tokens
    tpm: int | None = None  # tokens per minute
    rpm: int | None = None  # requests per minute
    base_url: str | None = None  # API base URL (if applicable)
    output_format: Literal['json_schema', 'json_object', 'json_text' ] = 'json_schema'  # 出力フォーマットの指定方法

class AIProvider(BaseModel, frozen=True):
    name: str # e.g. "CPU", "OpenAI", "Anthropic", "Gemini"
    models: list[AIModel] = [] # e.g. [ "gpt-4o", "gpt-4o-mini" ]
    default_index:int = 0

    def to_model_or_default(self,name:str|None,default:str|None=None) -> str:
        if name:
            for p in self.models:
                if p.name.lower() == name.lower() or p.model.lower() == name.lower():
                    return p.model
        return default or self.models[0].model

    def get_model(self, name:str|None, config:LLMBaseConfig|None=None) -> AIModel|None:
        if not name and config:
            name = config.model
        if not name:
            return None
        for p in self.models:
            if p.name.lower() == name.lower() or p.model.lower() == name.lower():
                return p
        return None

    def default(self) -> AIModel:
        if self.default_index>0:
            return self.models[self.default_index]
        return self.models[0]

    def find(self, name:str) -> AIModel|None:
        if name:
            for p in self.models:
                if p.name.lower() == name.lower() or p.model.lower() == name.lower():
                    return p
        return None

class AIListResponse(BaseModel):
    providers: List[AIProvider] = []

class MatchStateResponse(BaseModel):
    match_id: str
    turn: int
    status: MatchStatus


class PayloadUnit(BaseModel):
    """UnitHolder.to_payload 用のPydanticモデル。
    自軍/敵軍いずれでも利用できる最小公倍のサマリ構造。
    """
    id: str
    hp: int
    max_hp: int
    # 位置（非アクティブ時や敵推定時は None）
    x: Optional[int] = None
    y: Optional[int] = None
    # そのターン内の移動開始位置（見えていれば）
    x0: Optional[int] = None
    y0: Optional[int] = None
    # 自軍向けの詳細（敵側では通常 None）
    vision: Optional[int] = None
    speed: Optional[int] = None
    fuel: Optional[int] = None
    state: Optional[Literal['onboard', "outbound", "engaging", "returning", "lost"]] = None
    target: Optional[Position] = None

class SideViewPayload(BaseModel):
    """1視点（自軍/敵軍）における公開情報の入れ物"""
    carrier: Optional[PayloadUnit] = None
    squadrons: Optional[List[PayloadUnit]] = None
    turn_visible: Optional[List[str]] = None


# --- Realtime/SSE payload models ---
# クライアントに配信する詳細な状態（SSE向け）。

class TurnLog(BaseModel):
    step:int
    unit_id: str
    unit_pos: Position|None = None
    report: Literal["target", "returning", "attack", "hit", "lost", "landed", "engaging", "found"]
    target_id: str|None
    target_pos: Position|None = None
    target_from: Position|None = None
    value: int|None = None

    def __init__(self, step:int, unit_id:str, unit_pos:Position|None, report:Literal["target", "returning", "attack", "hit", "lost", "landed", "engaging", "found"], *, value:int|None=None, target_id:str|None=None, target_pos:Position|None=None, target_from:Position|None=None):
        super().__init__(step=step, unit_id=unit_id, report=report, value=value, unit_pos=unit_pos, target_id=target_id, target_pos=target_pos, target_from=target_from)


WaitingFor = Literal["none", "orders", "you", "opponent"]

class MatchStatePayload(BaseModel):
    type: Literal["state"] = "state"
    match_id: str
    status: MatchStatus
    turn: int
    max_turn: int
    waiting_for: WaitingFor = "none"
    map_w: int
    map_h: int
    map: Optional[List[List[int]]] = None  # タイル情報（0:海, 1:陸地)
    # 自軍/敵軍ビュー
    units: SideViewPayload
    intel: SideViewPayload
    # 直近ターンのログ（視点別）
    logs: Optional[List[TurnLog]] = None
    # 結果（視点別 win/lose/draw）
    result: Optional[Literal["win", "lose", "draw"]] = None
    # LLM簡易診断（PvE・人間視点のみ）。注文JSONは含めない。
    ai_diag: Optional[Dict[str, Any]] = None
    # 画面上部表示用の便宜フィールド（視点別）
    you_name: Optional[str] = None
    opponent_name: Optional[str] = None


class MatchOrdersRequest(BaseModel):
    player_token: str
    player_orders: Optional[PlayerOrders] = None
    # 将来: readyフラグ/キャンセル等


class MatchOrdersResponse(BaseModel):
    turn: int
    accepted: bool
    status: MatchStatus
    logs: List[str] = []
