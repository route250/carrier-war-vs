
import time
from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue
from server.schemas import MatchJoinRequest, MatchJoinResponse, MatchStatePayload, PlayerOrders, MatchStateResponse, MatchOrdersRequest, MatchOrdersResponse
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from server.services.match import MatchStore

class AIStat(Enum):
    """AIの状態を表す列挙型"""
    NOT_START = 0
    STARTING = 1
    READY = 2
    CLEANUP = 4
    STOPPING = 3
    STOPPED = 5
    ERROR = 9

"""AIの思考ルーチンを実装するための抽象クラスと具体的なクラス"""
class AIThreadABC(ABC):
    """AIの思考ルーチンを実装するための抽象クラス"""
    def __init__(self, store:'MatchStore', match_id: str):
        self.store: MatchStore = store
        self.match_id = match_id
        self.token = None
        self.side = None
        self.q: Queue[MatchStatePayload|None] = Queue() # タイミング通知用のキュー（自身のイベントループ専用）
        self.stat:AIStat = AIStat.NOT_START
        self.maparray:list[list[int]] = []
        self._last_turn:int = -1

    @property
    def name(self) -> str:
        return f"{self.__class__.name}"

    def is_alive(self) -> bool:
        if self.stat == AIStat.STARTING:
            return True
        if self.stat == AIStat.READY:
            if self.token and self.side:
                return True
            self.stat = AIStat.ERROR
        return False

    def is_ready(self) -> bool:
        return self.is_alive() and self.stat == AIStat.READY

    def is_match_waiting(self) -> bool:
        return self.store.is_match_waiting(self.match_id)

    def is_match_active(self) -> bool:
        return self.store.is_match_active(self.match_id)

    def stop(self):
        if self.stat == AIStat.NOT_START:
            self.stat = AIStat.STOPPED
            return
        if self.stat == AIStat.READY or self.stat == AIStat.STARTING :
            self.stat = AIStat.STOPPING
            self.q.queue.clear()
            self.q.put_nowait(None)

    def put_payload(self, payload:MatchStatePayload):
        if not self.is_alive():
            self.stop()
            return
        self.q.put_nowait(payload)

    def run(self):
        self.stat = AIStat.STARTING
        join_req:MatchJoinRequest = MatchJoinRequest(display_name=self.name)
        join_res: MatchJoinResponse = self.store.join(self.match_id, join_req)
        if not join_res.player_token:
            self.stat = AIStat.ERROR
            return
        try:
            self.token = join_res.player_token
            self.side = join_res.side
            while self.is_match_waiting():
                time.sleep(1.0)
                if self.stat!=AIStat.STARTING:
                    return
            self.maparray = self.store.get_map_array(self.match_id)
            snap = self.store.snapshot(self.match_id, token=self.token)
            self.max_turn = snap.max_turn
            snap.units.turn_visible = None  # AIには視界情報は不要
            self.q.put_nowait(snap)
            self.stat = AIStat.READY

            self._dbg_print(0, "loop startup")
            self.startup()

            while self.is_alive():
                # 次のターンを待つ
                payload = self.q.get()
                if payload is None:
                    self._dbg_print(self._last_turn,f"[{self.side}] loop received None, ABORT")
                    self.abort(None)
                    break
                status = payload.status # None, "active", "waiting", "over"
                result = payload.result # None, "win", "lose", "draw"
                waiting_for = payload.waiting_for or "none" # "none", "orders", "you", "opponent"                
                # self._dbg_print(payload.turn,f"[{self.side}] loop received wait_for:{payload.waiting_for}")
                if not self.is_alive():
                    self._dbg_print(payload.turn,f"[{self.side}] loop not alive, ABORT")
                    break
                elif status and status == "waiting":
                    # self._dbg_print(payload.turn,f"[{self.side}] loop match waiting, CONTINUE")
                    continue
                elif not self.is_match_active():
                    self._dbg_print(payload.turn,f"[{self.side}] loop no match state, ABORT")
                    self.abort(payload)
                    break
                elif payload.turn <= self._last_turn:
                    # self._dbg_print(payload.turn,f"[{self.side}] loop received old turn:{payload.turn} <= {self._last_turn}, CONTINUE")
                    continue

                self._last_turn = payload.turn
                payload.map = None

                # 終了していたら抜ける

                if status == "over" or result is not None:
                    self._dbg_print(payload.turn,f"[{self.side}] loop game over {status}/{result}, ABORT")
                    self.over(payload)
                    break
                 # 自分のターンでなかったら、待つ
                if waiting_for != "orders" and waiting_for != "you":
                    self._dbg_print(payload.turn,f"[{self.side}] loop not your turn ({waiting_for}), CONTINUE")
                    continue
                # 思考して命令を出す
                self._dbg_print(payload.turn,f"[{self.side}] loop thinking enter...")
                orders = self.think(payload)
                self._dbg_print(payload.turn,f"[{self.side}] loop thinking exit...")
                # self.thinkの中でオーダーがエラーの場合のリトライはしているので、ここではエラーなら空オーダーにする。
                if not self.submit_orders(orders):
                    self.submit_orders(PlayerOrders())

            if self.stat == AIStat.READY or self.stat == AIStat.CLEANUP or self.stat == AIStat.STOPPING:
                self.stat = AIStat.CLEANUP
                self._dbg_print(self._last_turn,f"[{self.side}] cleanup")
                self.cleanup()
        except Exception as e:
            self.stat = AIStat.ERROR
            self._dbg_print(self._last_turn,f"[{self.side}] exception occurred: {e}")
        finally:
            self.stat = AIStat.STOPPING if self.stat != AIStat.ERROR else AIStat.ERROR
            # マッチから抜ける
            if self.token:
                self._dbg_print(self._last_turn,f"[{self.side}] leaving match {self.match_id}")
                self.store.leave(self.match_id, self.token)
            self.stat = AIStat.STOPPED if self.stat != AIStat.ERROR else AIStat.ERROR

    def submit_orders(self, orders: PlayerOrders) -> bool:
        if self.token is None:
            return False
        """思考ルーチンがオーダーを出して結果を受け取るための関数"""
        order_req = MatchOrdersRequest(player_token=self.token, player_orders=orders)
        order_res = self.store.submit_orders(self.match_id, order_req)
        return order_res.accepted

    def validate_orders(self, orders: PlayerOrders) -> list[str]:
        if self.token is None:
            return ["no token. can not continue. you abort all process."]
        """思考ルーチンがオーダーを出して結果を受け取るための関数"""
        order_req = MatchOrdersRequest(player_token=self.token, player_orders=orders)
        mesg = self.store.validate_orders(self.match_id, order_req)
        if mesg is None:
            return ["not accepted"]
        return mesg

    @abstractmethod
    def _ensure_client_ready(self) -> bool:
        """クライアント初期化の成否を返す。FalseならLLM未使用で待機。"""

    @abstractmethod
    def startup(self) -> None:
        """作戦終了後にレビューするための抽象メソッド"""

    @abstractmethod
    def abort(self, payload:MatchStatePayload|None): ...

    @abstractmethod
    def over(self, payload:MatchStatePayload|None): ...

    @abstractmethod
    def think(self, payload:MatchStatePayload) -> PlayerOrders:
        """思考ルーチンを実装するための抽象メソッド"""
        # AIの思考処理をここに実装
        # orderは、PlayerOrdersのインスタンスで away self.on_orders(orders) を呼び出して、メッセージがなければOK
        order = PlayerOrders()
        return order

    @abstractmethod
    def cleanup(self) -> None:
        """作戦終了後にレビューするための抽象メソッド"""

    def _dbg_print(self, turn: int, content: str) -> None:
        if not content:
            print()
        else:
            llm=self.name
            print(f"[{llm}][turn:{turn}] {content}")

class AIThreadEasy(AIThreadABC):
    """簡単な思考ルーチン"""
    def think(self, payload:MatchStatePayload) -> PlayerOrders:
        # 簡単な思考ルーチン
        order = PlayerOrders()
        return order

class AIThreadMedium(AIThreadABC):
    """普通の思考ルーチン"""
    def think(self, payload:MatchStatePayload) -> PlayerOrders:
        # 普通の思考ルーチン
        order = PlayerOrders()
        return order

class AIThreadHard(AIThreadABC):
    """難しい思考ルーチン"""
    def think(self, payload:MatchStatePayload) -> PlayerOrders:
        # 難しい思考ルーチン
        order = PlayerOrders()
        return order
