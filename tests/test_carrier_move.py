import unittest
from server.schemas import CarrierState, MatchCreateRequest, PlayerOrders, Position, SquadronState
from server.services.match import Match, MatchStore

import unittest
from server.schemas import MatchCreateRequest, Position
from server.services.match import MatchStore
from tests.utils import create_match, get_units, set_orders

class TestCarrierMoveAdvanced(unittest.TestCase):
    def setUp(self):
        store, match = create_match()
        self.store = store
        self.match: Match = match
        carrier, squadrons = get_units(self.match, 'A')
        assert carrier is not None, "carrierがNoneです"
        self.a_token = match.side_a.token
        self.a_carrier = carrier
        self.a_squadrons = squadrons
        self.init_pos = Position(x=self.a_carrier.pos.x, y=self.a_carrier.pos.y)
        carrier, squadrons = get_units(self.match, 'B')
        self.b_token = match.side_a.token
        self.b_carrier = carrier
        self.b_squadrons = squadrons

    def set_orders(self, side:str, orders: PlayerOrders ):
        return set_orders( self.store, self.match, side, orders )

    def test_no_order_no_move(self):
        # 1. オーダーなしで移動しない
        self.set_orders( 'A', PlayerOrders() )
        self.set_orders( 'B', PlayerOrders() )
        self.match._resolve_turn_minimal()
        # 空母が移動していないこと
        self.assertEqual( self.a_carrier.pos, self.init_pos )

    def test_multi_step_to_target(self):
        # 3. オーダー無しで目標地点まで複数回移動
        target = Position(x=self.match.map.W-3, y=self.match.map.H//2)
        self.set_orders( 'A', PlayerOrders(carrier_target=target) )
        self.set_orders( 'B', PlayerOrders() )

        self.match.map.max_turn=99999

        reached = False
        max_turns = 30
        prev_pos = self.a_carrier.pos
        for _ in range(max_turns):
            self.match._resolve_turn_minimal()
            self.assertNotEqual( prev_pos, self.a_carrier.pos, "空母が移動しませんでした")
            if self.a_carrier.pos == target:
                reached = True
                break
        self.assertTrue(reached, f"キャリアが{max_turns}ターン以内に目標に到達しませんでした: pos={(self.a_carrier.pos.x,self.a_carrier.pos.y)}")

        # さらにターン進行しても動かない
        self.set_orders( 'A', PlayerOrders() )
        self.set_orders( 'B', PlayerOrders() )
        for _ in range(max_turns):
            self.match._resolve_turn_minimal()
            self.assertEqual( self.a_carrier.pos, target, "オーダしてないのに空母が動きました")

        # 3. オーダー無しで目標地点まで複数回移動
        target = Position(x=3, y=self.match.map.H//2)
        self.set_orders( 'A', PlayerOrders(carrier_target=target) )
        self.set_orders( 'B', PlayerOrders() )
        for _ in range(3):
            self.match._resolve_turn_minimal()
            self.assertNotEqual( prev_pos, self.a_carrier.pos, "空母が移動しませんでした")
        target = Position(x=3, y=3)
        self.set_orders( 'A', PlayerOrders(carrier_target=target) )
        self.set_orders( 'B', PlayerOrders() )
        for _ in range(max_turns):
            self.match._resolve_turn_minimal()
            self.assertNotEqual( prev_pos, self.a_carrier.pos, "空母が移動しませんでした")
            if self.a_carrier.pos == target:
                reached = True
                break
        self.assertEqual(self.a_carrier.pos, target, "途中変更で目的地に到達しませんでした。")

    def test_squadron_launch_and_return(self):
        # 6. 航空機を発艦させ、目標到達後に戻ってくることを確認する
        ctarget = Position(x=self.match.map.W-3, y=self.match.map.H//2)
        starget = Position(x=self.match.map.W//2, y=self.match.map.H//2)

        self.set_orders( 'A', PlayerOrders(carrier_target=ctarget,launch_target=starget) )
        self.set_orders( 'B', PlayerOrders() )
        
        max_turns = 30
        creached = False
        sreached = False
        sreturned = False
        for _ in range(max_turns):
            self.match._resolve_turn_minimal()
            if self.a_carrier.pos == ctarget:
                creached = True
            for s in self.a_squadrons:
                if s.pos == starget:
                    sreached = True
                    break
            if creached and sreached:
                x = next( (s for s in self.a_squadrons if s.state != 'onboard'), None)
                if x is None:
                    sreturned = True
                    break

        self.assertTrue(creached, f"キャリアが{max_turns}ターン以内に目標に到達しませんでした: pos={(self.a_carrier.pos.x,self.a_carrier.pos.y)}")
        self.assertTrue(sreached, f"艦載機が{max_turns}ターン以内に目標に到達しませんでした")
        self.assertTrue(sreturned, f"艦載機が{max_turns}ターン以内に基地に戻りませんでした")

if __name__ == '__main__':
    unittest.main()
