

from server.schemas import CarrierState, MatchCreateRequest, MatchJoinRequest, MatchOrdersRequest, MatchOrdersResponse, MatchStatus, PlayerOrders, Position, SquadronState
from server.services.match import Match, MatchStore
from server.schemas import CarrierState, MatchCreateRequest, SquadronState
from server.services.match import Match, MatchStore
from server.services.hexmap import HexArray, generate_connected_map


def get_units(match: Match, side: str) -> tuple[CarrierState | None, list[SquadronState]]:
    carrier = next((s.unit for s in match.map.units_list if s.side == side and isinstance(s.unit, CarrierState)), None)
    squadrons = [s.unit for s in match.map.units_list if s.side == side and isinstance(s.unit, SquadronState)]
    return carrier, squadrons

def create_match() -> tuple[MatchStore, Match]:
    print("Setup for tests")
    store = MatchStore()
    req = MatchCreateRequest(mode="pvp", config=None, display_name="test")
    resp = store.create(req)
    match: Match = store._matches[resp.match_id]

    p1 = Position(x=3,y=3)
    p2 = Position(x=match.map.W-3,y=match.map.H-3)
    for u in match.map.units_list:
        if isinstance(u.unit, CarrierState) and u.side=='A':
            u.unit.pos = p1
        if isinstance(u.unit, CarrierState) and u.side=='B':
            u.unit.pos = p2
    #generate_connected_map(match.map.hexmap, excludes=[p1,p2])
    for y in range(match.map.H):
        for x in range(match.map.W):
            match.map.hexmap.set(x,y,0)
    join_res = MatchJoinRequest( display_name='usrB' )
    store.join(match.match_id,join_res)
    return store, match

def set_orders(store:MatchStore, match:Match, side:str, orders:PlayerOrders ) -> MatchStatus:
    if side == 'A' and match.side_a.token:
        token: str = match.side_a.token
    elif side == 'B' and match.side_b.token:
        token: str = match.side_b.token
    else:
        raise ValueError(f"invalid side:{side}")
    req: MatchOrdersRequest = MatchOrdersRequest(player_token=token, player_orders=orders)
    res: MatchOrdersResponse = store.submit_orders(match.match_id, req)
    if not res.accepted:
        raise ValueError('\n'.join(res.logs) )
    return res.status
