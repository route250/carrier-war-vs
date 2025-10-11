
from dataclasses import dataclass, field
from server.schemas import PlayerOrders
from server.schemas import Position, UnitState, CarrierState, SquadronState, PayloadUnit, SideViewPayload, TurnLog
from server.schemas import SQUAD_MAX_HP, CARRIER_MAX_HP
from server.services.hexmap import HexArray
from server.utils.audit import match_write
import random
import os
import sys

# Debug flag: enable when running tests or when env var CARRIER_WAR_DEBUG is set
DEBUG = bool(os.getenv('CARRIER_WAR_DEBUG')) or ('unittest' in sys.modules) or ('PYTEST_CURRENT_TEST' in os.environ)

def _dbg(log_id: str | None, *args, **kwargs):
    """Debug helper: prints when DEBUG, always writes to match log.

    - print: 環境変数/テスト時のみ
    - file: `match_write` へ `{"type":"debug","msg":...}` を常に出力（best-effort）
    """
    if DEBUG:
        print(*args, **kwargs)
    try:
        msg = " ".join(str(a) for a in args)
        match_write(log_id, {"type": "debug", "msg": msg})
    except Exception:
        pass

class UnitHolder:
    def __init__(self, side, unit: UnitState):
        self.side = side
        self.unit:UnitState = unit
        self.ticks:int = 0
        self.next_time:int = 0
        #
        self.path:list[Position] = [unit.pos] if unit.is_active() else [] # 移動履歴
        self.intel:dict[int,Position] = {}  # 敵に発見された時刻と位置(敵側への報告用)

    def reset(self):
        self.ticks = 0
        self.next_time = 0
        self.path = [self.unit.pos] if self.unit.is_active() else []
        self.intel = {}

    def to_payload(self) -> PayloadUnit:
        result = PayloadUnit(
            id=self.unit.id,
            hp=self.unit.hp,
            max_hp=self.unit.max_hp,
            vision=self.unit.vision,
            speed=self.unit.speed,
            fuel=self.unit.fuel,
            state=(self.unit.state if isinstance(self.unit, SquadronState) else None),
            x=(self.unit.pos.x if self.unit.is_active() else None),
            y=(self.unit.pos.y if self.unit.is_active() else None),
        )
        if self.unit.is_active():
            if self.path:
                result.x0 = self.path[0].x
                result.y0 = self.path[0].y
            if self.unit.target:
                result.target = self.unit.target
        return result


    def to_turn_visible(self, side: str | None) -> set[Position]:
        result: set[Position] = set()
        if side is None or side == self.side:
            # ユニットが索敵した範囲のPosition集合を返す
            if self.unit.is_active() and self.path:
                r = self.unit.vision
                for p in self.path:
                    # pからr以内の位置を追加
                    for dx in range(-r, r+1):
                        for dy in range(-r, r+1):
                            pos = Position(x=p.x+dx, y=p.y+dy)
                            if p.hex_distance(pos) <= r:
                                result.add(pos)
        return result

def next_step_by_gradient( hexmap:HexArray, units: list[UnitHolder], current: Position, target: Position, *, ignore_land:bool = False, dist_array:list[list[int]]|None=None) -> Position|None:
    for pos in hexmap.neighbors_by_gradient(current, target, ignore_land=ignore_land, dist_array=dist_array):
        if all( not ou.unit.is_active() or pos != ou.unit.pos for ou in units):
            return pos
    return None

def next_step( hexmap:HexArray, units: list[UnitHolder], start: Position, current: Position, target: Position, *, ignore_land:bool = False, dist_array:list[list[int]]|None=None) -> Position|None:
    for pos in hexmap.neighbors_by_distance(start,current, target, ignore_land=ignore_land, dist_array=dist_array):
        if all( not ou.unit.is_active() or pos != ou.unit.pos for ou in units):
            return pos
    return None

# 攻撃判定: 編隊からのダメージと、空母からの対空(AA)
def scaled_damage(hp: int, max_hp:int, base: int) -> int:
    hp = hp if hp is not None else max_hp
    scale = max(0.0, min(1.0, hp / float(max_hp)))
    variance = round(base * 0.2)
    raw = base + (0 if variance == 0 else random.randint(-variance, variance))
    return max(0, round(raw * scale))

@dataclass
class IntelPath:
    """索敵結果"""
    side: str
    unit_id: str
    turn: int
    p1: Position
    p2: Position

@dataclass
class IntelReport:
    """索敵報告"""
    turn: int
    side: str
    logs: list[TurnLog] = field(default_factory=list)
    intel: dict[str,IntelPath] = field(default_factory=dict)

    def append(self, intel: dict[str,IntelPath]):
        self.intel.update(intel)

    def dump(self, board: 'GameBord'):
        yield f"side: {self.side} turn: {self.turn}"
        for log in self.logs:
            yield f"  log: {log}"
        for hld in board.units_list:
            unit = hld.unit
            if isinstance(unit,CarrierState):
                yield f"  unit: {unit.id} pos: {unit.pos} hp: {unit.hp}"
            elif isinstance(unit,SquadronState):
                loc = f"{unit.state}"
                if unit.pos.is_valid():
                    loc = loc + f"({unit.pos.x},{unit.pos.y})"
                yield f"  unit: {unit.id} {loc} hp: {unit.hp}"
        for path in self.intel.values():
            yield f"  intel: {path.unit_id} from {path.p1} to {path.p2}"

class GameBord:
    def __init__(self, hexmap: HexArray, units_list:list[list[UnitState]], *, log_id: str | None = None):
        if len(units_list) == 0:
            raise ValueError("Units list and orders list must have the same length.")
        if len(units_list) != 2:
            raise ValueError("This game only supports 2 players.")
        if hexmap is None:
            raise ValueError("Map cannot be None.")

        self.turn:int = 1
        self.max_turn:int = 30
        self.result:str|None = None
        self.first_spotting: str|None = None
        self.hexmap:HexArray = hexmap
        self.units_list:list[UnitHolder] = []
        self.log_id: str | None = log_id
        for side, bbb in zip(["A","B"], units_list):
            for unit in bbb:
                if isinstance(unit, CarrierState):
                    unit.pos = self.get_start_position(unit.pos) or unit.pos
                self.units_list.append(UnitHolder(side, unit))
        self.intel: dict[str,IntelReport] = {"A":IntelReport(side="A",turn=0), "B":IntelReport(side="B",turn=0)}
        _dbg(self.log_id, f"[GameBord] init W={self.W} H={self.H} units={len(self.units_list)}")
        # bootstrap log (best-effort)
        match_write(self.log_id, {
            "type": "match_bootstrap",
            "map_w": self.W,
            "map_h": self.H,
            "units": [
                {
                    "side": u.side,
                    "id": u.unit.id,
                    "type": ("carrier" if isinstance(u.unit, CarrierState) else "squadron"),
                    "pos": [u.unit.pos.x, u.unit.pos.y] if u.unit.is_active() else None,
                    "hp": u.unit.hp,
                }
                for u in self.units_list
            ],
        })

    @property
    def W(self) -> int:
        return self.hexmap.W
    @property
    def H(self) -> int:
        return self.hexmap.H

    def get_start_position(self, pos: Position) -> Position|None:
        """空母の初期位置をランダムに決定する。指定位置が有効ならばその位置を返す。"""
        # 候補地のリストを作成
        point_list: list[tuple[float,Position]] = []
        point_set: set[Position] = set()
        xrange1 = int(self.hexmap.W * 0.2)+1
        yrange1 = int(self.hexmap.H * 0.4)+1
        for dx in range(-xrange1, xrange1+1):
            for dy in range(-yrange1, yrange1+1):
                x = pos.x + dx
                y = pos.y + dy
                if 0 <= x < self.hexmap.W and 0 <= y < self.hexmap.H:
                    if self.hexmap.get(x,y) == 0:
                        p = Position(x=x,y=y)
                        if p not in point_set:
                            point_set.add(p)
                            d = pos.length(p)
                            point_list.append( (d,p) )
        xrange2 = int(self.hexmap.W * 0.4)+1
        yrange2 = int(self.hexmap.H * 0.2)+1
        for dx in range(-xrange2, xrange2+1):
            for dy in range(-yrange2, yrange2+1):
                x = pos.x + dx
                y = pos.y + dy
                if 0 <= x < self.hexmap.W and 0 <= y < self.hexmap.H:
                    if self.hexmap.get(x,y) == 0:
                        p = Position(x=x,y=y)
                        if p not in point_set:
                            point_set.add(p)
                            d = pos.length(p)
                            point_list.append( (d,p) )

        if not point_list:
            return None

        # 距離順にソート
        point_list.sort(key=lambda x: x[0])

        weights = [1.0 / (1 + dist) for dist, _ in point_list]
        total = sum(weights)
        r = random.random() * total
        acc = 0.0
        for weight, (_, candidate) in zip(weights, point_list):
            acc += weight
            if r <= acc:
                return candidate

        return point_list[-1][1]

    def get_map_array(self) -> list[list[int]]:
        return self.hexmap.copy_as_list()

    def adjust_target(self, origin_pos: Position, target_pos: Position, unit_range:int, ignore_island:bool ) -> Position|None:
        """目標位置が移動可能な位置でない場合、最も近い移動可能な位置を返す。"""
        if origin_pos == target_pos or unit_range<=0:
            return None
        # まず地図単体で調整
        adjust_pos = self.hexmap.adjust_target(origin_pos, target_pos, ignore_island=ignore_island)
        if adjust_pos is None:
            return None
        target_pos = adjust_pos
        # 次にユニットを実際に移動して航続距離限界を調べる
        pos = origin_pos
        hist:list[Position] = [pos]
        dist_array = self.hexmap.gradient_field(target_pos, ignore_land=ignore_island)
        for _ in range(min(self.hexmap.H*2+self.hexmap.W*2,unit_range)):
            npos = next_step(self.hexmap, self.units_list, origin_pos, pos, target_pos, ignore_land=ignore_island, dist_array=dist_array)
            if npos is None or npos in hist:
                break
            pos = npos
            hist.append(pos)
            hist = hist[-100:]
            if pos == target_pos:
                break
        return pos if pos != origin_pos else None


    def validate_orders(self, side:str, orders: PlayerOrders|None) -> tuple[list[str],list[str]]:
        msgs1: list[str] = []
        msgs2: list[str] = []
        if orders is None:
            return [],[]
        carrier = next((u for u in self.units_list if u.side == side and isinstance(u.unit, CarrierState)), None)
        squadron = next((u for u in self.units_list if u.side == side and isinstance(u.unit, SquadronState) and u.unit.state == 'onboard'), None)

        if orders.carrier_target is not None:
            if carrier is None or not carrier.unit.is_active():
                msgs1.append("Carrier not found or sunk.")
            else:
                errmsg = None
                if not (0 <= orders.carrier_target.x < self.hexmap.W and 0 <= orders.carrier_target.y < self.hexmap.H):
                    errmsg = f"Carrier target was rejected. {orders.carrier_target} out of map bounds."
                elif self.hexmap.get(orders.carrier_target.x, orders.carrier_target.y) != 0:
                    errmsg = f"Carrier target was rejected. {orders.carrier_target} is not on sea."
                if errmsg is not None:
                    adjust_pos = self.adjust_target(carrier.unit.pos, orders.carrier_target, carrier.unit.fuel, ignore_island=False)
                    if adjust_pos is not None:
                        errmsg += f" Please specify a valid coordinate, such as {adjust_pos}."
                    msgs1.append(errmsg)

        if orders.launch_target is not None:
            if squadron is None or carrier is None or not carrier.unit.is_active():
                msgs2.append("launch_target was rejected. Currently, all air squadrons are deployed on missions, and there are no squadrons aboard the carrier. Orders cannot be issued to squadrons that are already deployed.")
            elif orders.launch_target == carrier.unit.pos:
                msgs2.append("launch_target was rejected. Cannot be the same as carrier position.")
            else:
                errmsg = None
                if not (0 <= orders.launch_target.x < self.hexmap.W and 0 <= orders.launch_target.y < self.hexmap.H):
                    errmsg=f"launch_target was rejected. {orders.launch_target} is out of map bounds."
                elif carrier.unit.pos.hex_distance(orders.launch_target) > squadron.unit.fuel:
                    errmsg=f"launch_target was rejected. {orders.launch_target} is out of squadron range."
                if errmsg is not None:
                    adjust_pos = self.adjust_target(carrier.unit.pos, orders.launch_target, squadron.unit.fuel, ignore_island=True)
                    if adjust_pos is not None:
                        errmsg += f" Please specify a valid coordinate, such as {adjust_pos}."
                    msgs2.append(errmsg)
        return msgs1,msgs2

    def to_payload(self, view_side:str|None=None) -> tuple[SideViewPayload,SideViewPayload]:
        side = view_side if view_side in ['A','B'] else 'A'
        other_intel = self.intel.get(side) or IntelReport(side=side, turn=self.turn)
        my_carrier = None
        my_squadrons = []
        other_carrier = None
        other_squadrons = []
        my_turn_visible: set[Position] = set()
        for u in self.units_list:
            if side == u.side:
                pd = u.to_payload()
                if isinstance(u.unit, CarrierState):
                    my_carrier = pd
                elif isinstance(u.unit, SquadronState):
                    my_squadrons.append(pd)
            elif self.is_over():
                # ゲーム終了後は全てのユニットを表示
                pd = u.to_payload()
                if isinstance(u.unit, CarrierState):
                    other_carrier = pd
                elif isinstance(u.unit, SquadronState):
                    other_squadrons.append(pd)
            else:
                ir_path = other_intel.intel.get(u.unit.id)
                if ir_path:
                    if isinstance(u.unit, CarrierState):
                        other_carrier = PayloadUnit( id=u.unit.id,
                            hp=u.unit.hp, max_hp=u.unit.max_hp,
                            x=ir_path.p2.x, y=ir_path.p2.y,
                            x0=ir_path.p1.x, y0=ir_path.p1.y,
                        )
                    elif isinstance(u.unit, SquadronState):
                        other_squadrons.append(PayloadUnit( id=u.unit.id,
                            hp=u.unit.hp, max_hp=u.unit.max_hp,
                            x=ir_path.p2.x, y=ir_path.p2.y,
                            x0=ir_path.p1.x, y0=ir_path.p1.y,
                        ))
            tv = u.to_turn_visible(view_side)
            if side == u.side:
                for pos in tv:
                    if 0 <= pos.x < self.hexmap.W and 0 <= pos.y < self.hexmap.H:
                        my_turn_visible.add(pos)

        my_result = SideViewPayload()
        if my_carrier is not None:
            my_result.carrier = my_carrier
        if my_squadrons:
            my_result.squadrons = my_squadrons
        if my_turn_visible:
            vlist = sorted(list(my_turn_visible))
            my_result.turn_visible = [ f"{p.x},{p.y}" for p in vlist ]
        # 索敵結果をまとめる
        other_result = SideViewPayload()
        if other_carrier:
            other_result.carrier = other_carrier
        if other_squadrons:
            other_result.squadrons = other_squadrons
        return my_result, other_result

    def _get_carrier_by_side(self, side: str) -> tuple[int,int]:
        c = 0
        s = 0
        for u in self.units_list:
            if u.side == side:
                if isinstance(u.unit, CarrierState):
                    c += u.unit.hp if u.unit.hp is not None else 0
                elif isinstance(u.unit, SquadronState):
                    s += u.unit.hp if u.unit.hp is not None else 0
        return c,s

    def get_carrier_by_side(self, side: str) -> CarrierState|None:
        for u in self.units_list:
            if u.side == side:
                if isinstance(u.unit, CarrierState):
                    return u.unit
        return None

    def get_result(self) -> str|None:
        return self.result

    def get_squadrons_by_side(self, side: str) -> list[SquadronState]:
        return [u.unit for u in self.units_list if u.side == side and isinstance(u.unit, SquadronState)]

    def turn_forward(self, orders:list[PlayerOrders], intel_keep_turns:int = 3) -> dict[str,IntelReport]:
        logs: dict[str, list[TurnLog]] = {"A": [], "B": []}
        _dbg(self.log_id, f"[Turn {self.turn}] start")
        # file log: turn start
        match_write(self.log_id, {"type": "turn_start", "turn": self.turn})
        if len(orders) != 2:
            raise ValueError("Units list and orders list must have the same length.")

        # reset
        for u in self.units_list:
            u.reset()

        # orderのターゲット位置の妥当性チェックと設定
        for side, order in zip(["A","B"], orders):
            if order.carrier_target is not None:
                if not (0 <= order.carrier_target.x < self.hexmap.W and 0 <= order.carrier_target.y < self.hexmap.H):
                    raise ValueError(f"Carrier target {order.carrier_target} out of map bounds.")
                if self.hexmap.get(order.carrier_target.x, order.carrier_target.y) != 0:
                    raise ValueError(f"Carrier target {order.carrier_target} is not on sea.")
                for u in self.units_list:
                    if u.side == side and isinstance(u.unit, CarrierState):
                        u.unit.target = order.carrier_target
                        _dbg(self.log_id, f"[Turn {self.turn}] side {side} carrier target -> ({order.carrier_target.x},{order.carrier_target.y})")
                        match_write(self.log_id, {
                            "type": "order_carrier_target",
                            "turn": self.turn,
                            "side": side,
                            "target": [order.carrier_target.x, order.carrier_target.y],
                        })
                        break
        stp = 1
        # 判定フェーズ
        for u in self.units_list:
            if u.unit.is_active() and isinstance(u.unit, SquadronState) and u.unit.state=='engaging':
                # 攻撃中の航空部隊の攻撃処理に敵空母が居るか？
                ec = next( (cu for cu in self.units_list if cu.side != u.side and isinstance(cu.unit, CarrierState) and cu.unit.is_active() and u.unit.pos.hex_distance(cu.unit.pos) < 1.5), None)
                if ec:
                    u.ticks = u.unit.speed  # 攻撃完了まで動けない
                    ec.ticks = ec.unit.speed  # 攻撃完了まで動けない

                    aa = min( u.unit.hp, scaled_damage(ec.unit.hp,ec.unit.max_hp, 20) )
                    dmg = min( ec.unit.hp, scaled_damage(u.unit.hp,u.unit.max_hp, 25) )
                    # 空母へダメージ適用
                    u.unit.hp = u.unit.hp - aa
                    ec.unit.hp = ec.unit.hp - dmg

                    _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} attacks {ec.unit.id}: dmg={dmg}, AA={aa}")
                    match_write(self.log_id, {
                        "type": "attack",
                        "turn": self.turn,
                        "attacker": u.unit.id,
                        "defender": ec.unit.id,
                        "pos": [u.unit.pos.x, u.unit.pos.y],
                        "dmg_to_carrier": dmg,
                        "aa_to_attacker": aa,
                    })

                    # 空母側ログ
                    logs.setdefault(ec.side, []).append(TurnLog(stp, ec.unit.id, ec.unit.pos, "attack", target_id=u.unit.id, value=aa))
                    if ec.unit.hp <= 0:
                        # 撃沈
                        _dbg(self.log_id, f"[Turn {self.turn}] {ec.unit.id} sunk by {u.unit.id}")
                        ec.unit.target = None
                        ec.unit.pos = Position.invalid()
                        match_write(self.log_id, {"type": "sunk", "turn": self.turn, "unit": ec.unit.id, "by": u.unit.id})
                        logs.setdefault(ec.side, []).append(TurnLog(stp, ec.unit.id, ec.unit.pos, "lost", target_id=u.unit.id, value=dmg))
                    else:
                        logs.setdefault(ec.side, []).append(TurnLog(stp, ec.unit.id, ec.unit.pos, "hit", target_id=u.unit.id, value=dmg))

                    # 航空部隊側ログ
                    logs.setdefault(u.side, []).append(TurnLog(stp, u.unit.id, u.unit.pos, "attack", target_id=ec.unit.id, value=dmg))
                    if u.unit.hp <= 0:
                        # 撃墜
                        logs.setdefault(u.side, []).append(TurnLog(stp, u.unit.id, u.unit.pos, "lost", target_id=ec.unit.id, value=aa))
                        _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} shot down by AA")
                        u.unit.state = 'lost'
                        u.unit.pos = Position.invalid()
                        u.unit.target = None
                        match_write(self.log_id, {"type": "shot_down", "turn": self.turn, "unit": u.unit.id})
                    else:
                        logs.setdefault(u.side, []).append(TurnLog(stp, u.unit.id, u.unit.pos, "hit", target_id=ec.unit.id, value=aa))
                        logs.setdefault(u.side, []).append(TurnLog(stp, u.unit.id, u.unit.pos, "returning"))
                        _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} finished attack → returning")
                else:
                    logs.setdefault(u.side, []).append(TurnLog(stp, u.unit.id, u.unit.pos, "returning"))
                    _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} lost target → returning")
                # 攻撃完了したら帰還状態に変更
                u.unit.state = 'returning'

        #
        tick_queue:dict[int,list[UnitHolder]] = {}
        # 全ユニットの次の行動時間を設定
        for u in self.units_list:
            if u.unit.is_active() and u.ticks < u.unit.speed:
                u.next_time = int( 1000 / u.unit.speed )
                tick_queue.setdefault(u.next_time, []).append(u)

        # 移動と索敵ループ
        while tick_queue:
            current_time = min(tick_queue.keys())
            current_units = tick_queue.pop(current_time)
            random.shuffle(current_units)
            stp += 1
            # ユニットの移動フェーズ
            for u in current_units:
                if u.unit.origin is not None and u.unit.target is not None and u.unit.pos != u.unit.target and u.ticks < u.unit.speed:
                    if isinstance(u.unit, SquadronState) and u.unit.state == 'returning':
                        # 帰還中は空母の位置を目標にする
                        cu = next((cu for cu in self.units_list if cu.side == u.side and isinstance(cu.unit, CarrierState) and cu.unit.hp>0), None)
                        if cu is not None:
                            if cu.unit.pos.hex_distance(u.unit.pos) < 1.5:
                                # 空母に到達したら基地状態に変更
                                logs.setdefault(u.side, []).append(TurnLog(stp, u.unit.id, cu.unit.pos, "landed", target_id=cu.unit.id))
                                _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} returned to carrier {cu.unit.id}")
                                u.unit.state = 'onboard'
                                u.unit.pos = Position.invalid()
                                u.path.append(u.unit.pos)
                                u.unit.target = None
                                u.ticks = u.unit.speed
                                cu.ticks = cu.unit.speed # 着艦時には動けない
                                try:
                                    if self.log_id:
                                        match_write(self.log_id, {"type": "return", "turn": self.turn, "id": u.unit.id, "carrier": cu.unit.id})
                                except Exception:
                                    pass
                                continue
                            u.unit.target = cu.unit.pos
                    ignore_land = isinstance(u.unit, SquadronState)
                    next_pos = next_step(self.hexmap, self.units_list, u.unit.origin, u.unit.pos, u.unit.target, ignore_land=ignore_land)
                    if next_pos is not None:
                        if isinstance(u.unit, CarrierState):
                            _dbg(self.log_id, f"[Turn {self.turn}] carrier {u.unit.id} move {u.unit.pos.x},{u.unit.pos.y} -> {next_pos.x},{next_pos.y}")
                            match_write(self.log_id, {
                                "type": "move",
                                "turn": self.turn,
                                "unit": u.unit.id,
                                "from": [u.unit.pos.x, u.unit.pos.y],
                                "to": [next_pos.x, next_pos.y],
                            })
                        u.unit.pos = next_pos
                        u.path.append(u.unit.pos)
                        u.ticks += 1
                    if u.ticks < u.unit.speed:
                        u.next_time += int( 1000 / u.unit.speed )
                        tick_queue.setdefault(u.next_time, []).append(u)
            # 索敵フェーズ
            for u in self.units_list:
                for enemy in [ us for us in self.units_list if us is not u and us.side != u.side]:
                    if enemy.unit.is_active() and u.unit.can_see_enemy(enemy.unit):
                        if isinstance(enemy.unit, CarrierState) and self.first_spotting is None:
                            self.first_spotting = u.side
                            match_write(self.log_id, {"type": "first_spotting", "turn": self.turn, "side": u.side, "unit": u.unit.id, "enemy": enemy.unit.id})
                            _dbg(self.log_id, f"[Turn {self.turn}] First spotting by side {u.side} unit {u.unit.id}")

                        enemy.intel[current_time] = enemy.unit.pos
                        # 敵を発見したログ
                        xlog = logs.setdefault(u.side, [])
                        xlog[:] = [l for l in xlog if not (l.unit_id == u.unit.id and l.target_id == enemy.unit.id and l.report=="found")]
                        if not next( (l for l in xlog if l.unit_id == u.unit.id and l.target_id == enemy.unit.id), None):
                            xlog.append(TurnLog(stp, u.unit.id, u.unit.pos, "found", target_id=enemy.unit.id, target_pos=enemy.unit.pos))
                        _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} found {enemy.unit.id} at {enemy.unit.pos.x},{enemy.unit.pos.y}")
                        match_write(self.log_id, {
                            "type": "detect",
                            "turn": self.turn,
                            "by": u.unit.id,
                            "enemy": enemy.unit.id,
                            "pos": [enemy.unit.pos.x, enemy.unit.pos.y],
                        })
                        # 航空部隊が進出中に敵空母を発見したら攻撃目標に設定
                        if isinstance(u.unit, SquadronState) and u.unit.state=='outbound':
                            if isinstance(enemy.unit, CarrierState):
                                xlog = logs.setdefault(u.side, [])
                                xlog[:] = [l for l in xlog if not (l.unit_id == u.unit.id and l.target_id == enemy.unit.id and (l.report=="found" or l.report=="target"))]
                                xlog.append(TurnLog(stp, u.unit.id, u.unit.pos, "target", target_id=enemy.unit.id, target_pos=enemy.unit.pos))
                                u.unit.target = enemy.unit.pos
                                _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} switches target to enemy carrier {enemy.unit.id}")
        stp += 1
        # 判定フェーズ
        for u in self.units_list:
            if u.unit.is_active() and isinstance(u.unit, SquadronState) and u.unit.state=='outbound':
                # 攻撃中の航空部隊の攻撃処理に敵空母が居るか？
                ec = next( (cu for cu in self.units_list if cu.side != u.side and isinstance(cu.unit, CarrierState) and cu.unit.is_active() and u.unit.pos.hex_distance(cu.unit.pos) < 1.5), None)
                if ec:
                    xlog = logs.setdefault(u.side, [])
                    xlog[:] = [l for l in xlog if not (l.unit_id == u.unit.id and l.target_id == enemy.unit.id and (l.report=="found" or l.report=="target"))]
                    xlog.append(TurnLog(stp, u.unit.id, u.unit.pos, "engaging", target_id=ec.unit.id))
                    u.unit.state = 'engaging'
                    _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} starts attacking {ec.unit.id}")
                    match_write(self.log_id, {"type": "engage", "turn": self.turn, "attacker": u.unit.id, "defender": ec.unit.id})
                elif u.unit.pos == u.unit.target:
                    # 目標に到達したら帰還状態に変更
                    cu = next((cu for cu in self.units_list if cu.side == u.side and isinstance(cu.unit, CarrierState)), None)
                    if cu is not None:
                        u.unit.target = cu.unit.pos
                    else:
                        u.unit.target = u.unit.origin
                    logs.setdefault(u.side, []).append(TurnLog(stp, u.unit.id, u.unit.pos, "returning"))
                    u.unit.state = 'returning'
                    _dbg(self.log_id, f"[Turn {self.turn}] {u.unit.id} reached target → returning")
        # 発艦処理
        for side, order in zip(["A","B"], orders):
            if order.launch_target is not None:
                if not (0 <= order.launch_target.x < self.hexmap.W and 0 <= order.launch_target.y < self.hexmap.H):
                    raise ValueError(f"Launch target {order.launch_target} out of map bounds.")
                launched = False
                for u in self.units_list:
                    if u.side == side and isinstance(u.unit, SquadronState):
                        if u.unit.state == 'onboard':
                            # 空母の位置を取得
                            launch_pos = next((cu.unit.pos for cu in self.units_list if cu.side == side and isinstance(cu.unit, CarrierState)), None)
                            if launch_pos is not None:
                                # ターゲットに近い位置に発艦
                                pos = next_step(self.hexmap, self.units_list, launch_pos, launch_pos, order.launch_target,ignore_land=True)
                                if pos is not None:
                                    logs.setdefault(u.side, []).append(TurnLog(stp,u.unit.id, pos, "target", target_pos=order.launch_target))
                                    _dbg(self.log_id, f"[Turn {self.turn}] side {side} {u.unit.id} launched toward {order.launch_target.x},{order.launch_target.y}")
                                    match_write(self.log_id, {
                                        "type": "launch",
                                        "turn": self.turn,
                                        "side": side,
                                        "id": u.unit.id,
                                        "from": [launch_pos.x, launch_pos.y],
                                        "target": [order.launch_target.x, order.launch_target.y],
                                    })
                                    u.unit.pos = pos
                                    u.path.append(u.unit.pos)
                                    u.unit.state = 'outbound'
                                    u.unit.target = order.launch_target
                                    launched = True
                                    break
                    if launched:
                        break
        # ----
        for i, side in enumerate(["A","B"]):
            report = IntelReport(side=side, turn=self.turn)
            report.logs = logs.get(side, [])
            # 以前の情報を引き継ぐ
            prev_side = self.intel.get(side)
            # 索敵結果
            for u in self.units_list:
                if u.side != side and u.unit.is_active():
                    pos_list = [p for t,p in sorted(u.intel.items())] if u.intel else []
                    prev_ir_path = prev_side.intel.get(u.unit.id) if prev_side else None
                    ir_path: IntelPath|None = None
                    if pos_list:
                        if prev_ir_path and prev_ir_path.turn >= self.turn - intel_keep_turns:
                            pos_list.insert(0, prev_ir_path.p1)
                            pos_list.insert(1, prev_ir_path.p2)
                        ir_path = IntelPath( side=u.side, unit_id=u.unit.id, turn=self.turn, p1=pos_list[0], p2=pos_list[-1])
                        report.intel[u.unit.id] = ir_path
                    else:
                        if prev_ir_path and prev_ir_path.turn >= self.turn - intel_keep_turns:
                            report.intel[u.unit.id] = prev_ir_path

            self.intel[side] = report
        # ターン終了サマリ
        a_car = self.get_carrier_by_side("A")
        b_car = self.get_carrier_by_side("B")
        _dbg(self.log_id,
            f"[Turn {self.turn}] end: A({a_car.pos.x if a_car else None},{a_car.pos.y if a_car else None}) HP={a_car.hp if a_car else None} / "
            f"B({b_car.pos.x if b_car else None},{b_car.pos.y if b_car else None}) HP={b_car.hp if b_car else None}"
        )
        match_write(self.log_id, {
            "type": "turn_end",
            "turn": self.turn,
            "a": {"pos": ([a_car.pos.x, a_car.pos.y] if a_car else None), "hp": (a_car.hp if a_car else None)},
            "b": {"pos": ([b_car.pos.x, b_car.pos.y] if b_car else None), "hp": (b_car.hp if b_car else None)},
        })

        # 終了判定
        a_carrier, a_squadrons = self._get_carrier_by_side("A")
        b_carrier, b_squadrons = self._get_carrier_by_side("B")
        if a_carrier <= 0 and b_carrier <= 0:
            self.result = self.first_spotting or "draw"
        elif b_carrier <= 0:
            self.result = "A"
        elif a_carrier <= 0:
            self.result = "B"
        elif a_squadrons > 0 and b_squadrons <= 0:
            self.result = "A"
        elif b_squadrons > 0 and a_squadrons <= 0:
            self.result = "B"
        elif self.turn >= self.max_turn:
            if a_carrier > b_carrier:
                self.result = "A"
            elif b_carrier > a_carrier:
                self.result = "B"
            else:
                self.result = self.first_spotting or "draw"
        if self.result is None:
            self.turn += 1
        return self.intel

    def is_over(self) -> bool:
        return self.result is not None
