import pytest
from pathlib import Path
from server.services.turn import GameBord
from server.services.hexmap import HexArray, generate_connected_map
from server.schemas import Position, INF


def test_get_and_getitem():
    """マップの座標に対応する値が正しく取得できるか？"""
    h = HexArray(3, 2)
    h.set(2, 1, 5)
    pos = Position(x=2, y=1)
    assert h.get(2,1) == 5
    assert h[pos] == 5
    with pytest.raises(TypeError):
        _ = h[(1, 1)] # type: ignore


def test_distance_and_hex_distance():
    """Positionの距離計算のチェック"""
    a = Position(x=0, y=0)
    b = Position(x=2, y=0)
    h = HexArray(5, 5)
    center = Position(x=2, y=2)
    for pos in h.neighbors(center):
        print(f"hex_distance {pos.x},{pos.y}")
        assert h.distance(center, pos) == 1
        assert center.hex_distance(pos) == 1
        assert pos.hex_distance(center) == 1
    test_case = [
        [1, 1, 1], [1,2, 1], [1,3, 1], [2,1, 1], [2,3, 1],
        (0,0, 3), (1,0, 2), (2,0, 2), (3,0, 2), (4,0, 3),
        (0,4, 3), (1,4, 2), (2,4, 2), (3,4, 2), (4,4, -1),
        (0,1, 2), (0,2, 2), (0,3, 2),
        (4,1, 3), (4,2, 2), (4,3, 3),
    ]
    for x, y, actual in test_case:
        pos = Position(x=x, y=y)
        d = actual if actual>=0 else center.hex_distance(pos)
        print(f"hex_distance {x},{y} = {d}")
        assert h.distance(center, pos) == d
        assert center.hex_distance(pos) == d
        assert pos.hex_distance(center) == d


def test_gradient_field_and_path_basic():
    """基本的な勾配場と経路探索のテスト"""
    # 5x5 全て海のマップ
    h = HexArray(5, 5)
    goal = Position(x=2, y=2)
    dist = h.gradient_field(goal)
    # 目標地点は0
    assert dist[2][2] == 0
    # 各地点の距離はPosition.hex_distanceと一致する
    for x in range(h.W):
        for y in range(h.H):
            p = Position(x=x, y=y)
            d = p.hex_distance(goal)
            assert dist[y][x] == d
    for x in range(h.W):
        for y in range(h.H):
            start = Position(x=x, y=y)
            path = h.path_by_gradient(start, goal)
            if start==goal:
                assert len(path)==1 and path[0]==goal
                continue
            assert path and len(path)>1
            p0 = start
            d0 = 0
            for p in path:
                # 経路上の各点は前の点から距離1である
                assert p0.hex_distance(p) == 1 or start == p
                # 経路上の各点は目標からの距離が1ずつ減少する
                d = p.hex_distance(goal)
                assert d == d0-1 or start == p
                p0 = p
                d0 = d
            # 経路の最後は目標地点である
            assert path[-1].hex_distance(goal) == 0

def test_validate_carrier_path():
    """目的地に辿り着けないときに、正しい座標に補正できるか？"""
    print("test_validate_path")
    # 5x5 全て海のマップ
    h = HexArray(11, 11)
    mapdata = [
        [0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0],
         [0,0,0,0,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,1,0,0,0,0],
         [0,0,1,1,1,1,0,1,1,0,0],
        [0,0,0,1,1,0,0,1,0,0,0],
         [0,0,0,0,0,1,1,0,0,0,0],
        [0,0,0,0,0,1,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0],
    ]
    h.set_map(mapdata)
    # 各地点からの勾配場が海だけを計算しているか？
    dist = h.gradient_field(Position(x=0, y=0), ignore_land=False)
    for sx in range(h.W):
        for sy in range(h.H):
            if h.get(sx,sy) == 0:
                assert dist[sy][sx] != INF
            else:
                assert dist[sy][sx] == INF
    # 各地点から各地点への経路探索を試みる
    for sx in range(h.W):
        for sy in range(h.H):
            if 0<sx<h.W-1 and 0<sy<h.H-1:
                continue
            # 開始地点を設定
            start = Position(x=sx, y=sy)
            for gx in range(h.W):
                for gy in range(h.H):
                    if h.get(gx,gy) == 0:
                        continue
                    # 目標地点を設定
                    orig_goal = Position(x=gx, y=gy)
                    # 目的地が陸のはずなので、最も近い海に補正する
                    goal = h.adjust_target( start, orig_goal, ignore_island=False)
                    assert goal and goal != orig_goal
                    print(f"adjust target ({sx},{sy}) to ({gx},{gy}): ({goal.x},{goal.y})")
                    assert h.get(goal.x, goal.y) == 0
                    # 経路探索を実行
                    path = h.path_by_gradient(start, goal,ignore_land=False)
                    assert path and len(path)>0
                    print("validate path ({},{}) to ({},{}): {}".format(
                        sx, sy, goal.x, goal.y,
                        " ".join(f"({p.x},{p.y})" for p in path)
                    ))
                    assert path and len(path)>1
                    # 距離再計算
                    dist = h.gradient_field(goal, ignore_land=False)
                    # 経路の各点を検証
                    p0 = start
                    d0 = 0
                    for p in path:
                        # 海上を進むこと
                        assert h.get(p.x, p.y) == 0, f"経路が海じゃないよ: {p.x},{p.y}"
                        # 経路上の各点は前の点から距離1である
                        assert p0.hex_distance(p) == 1 or start == p, f"経路が飛んでるよ: {p0.x},{p0.y} -> {p.x},{p.y}"
                        # 経路上の各点は目標からの距離が1ずつ減少する
                        d = dist[p.y][p.x]
                        assert d < d0 or start == p, f"経路が遠ざかってるよ: {p.x},{p.y} (d={d} prev={d0})"
                        p0 = p
                        d0 = d
                    # 経路の最後は目標地点である
                    assert path[-1].hex_distance(goal) == 0, f"経路の最後が目標じゃないよ: {path[-1].x},{path[-1].y} != {goal.x},{goal.y}"


def test_validate_sq_path():
    """目的地に辿り着けないときに、正しい座標に補正できるか？"""
    print("test_validate_path")
    # 5x5 全て海のマップ
    h = HexArray(30,30)
    boad = GameBord(hexmap=h, units_list=[[],[]])

    # 各地点から各地点への経路探索を試みる
    start_pos = Position(x=h.W-1,y=h.H-1)
    fuel = 22
    for gx in range(h.W):
        for gy in range(h.H):
            # 目標地点を設定
            orig_goal = Position(x=gx, y=gy)
            orig_dist = start_pos.hex_distance(orig_goal)
            # 目的地が陸のはずなので、最も近い海に補正する
            goal = boad.adjust_target( start_pos, orig_goal, fuel, ignore_island=True)
            if goal:
                dist = start_pos.hex_distance(goal)
                print(f"adjust target ({start_pos.x},{start_pos.y}) to ({orig_goal.x},{orig_goal.y}) dist {orig_dist}: ({goal.x},{goal.y}) dist {dist}")
                if orig_dist<=fuel:
                    assert orig_goal == goal
                else:
                    assert orig_goal != goal
                    assert dist <= fuel
            else:
                print(f"adjust target ({start_pos.x},{start_pos.y}) to ({orig_goal.x},{orig_goal.y}) dist {orig_dist}: None")
                assert start_pos == orig_goal
                assert orig_dist == 0

def test_validate_sea_connectivity():
    # 手で陸を作って海が分断するケース
    h = HexArray(5, 5)
    # 横に陸の壁を作る
    for x in range(h.W):
        h.set(x,2,1)
    assert not h.validate_sea_connectivity()
    h.set(2,2,0)
    assert h.validate_sea_connectivity()

def test_generate_connected_map():
    h = HexArray(5, 7)
    assert 5 == h.W
    assert 7 == h.H
    generate_connected_map(h, blobs=5, seed=42)
    assert h.validate_sea_connectivity()

def test_neighbors_by_gradient_ordering():
    h = HexArray(5, 5)
    # place a simple target
    goal = Position(x=4, y=2)
    start = Position(x=2, y=2)
    nbrs = h.neighbors_by_gradient(start, goal)
    # 6 neighbors returned (some may be out of bounds filtered)
    assert isinstance(nbrs, list)
    if nbrs:
        # best neighbor should be closer to goal than start
        assert nbrs[0].hex_distance(goal) <= start.hex_distance(goal)


def test_find_path_respects_obstacles():
    h = HexArray(5, 5)
    # place a wall blocking direct path
    for x in range(1, h.W-1):
        h.set(x,2,1)
    start = Position(x=0, y=2)
    goal = Position(x=4, y=2)
    # a path around the obstacle should exist (via neighboring rows)
    path = h.find_path(start, goal)
    assert path is not None
    assert path[0] == start
    # allow ignoring land (also should find a path)
    path2 = h.find_path(start, goal, ignore_land=True)
    assert path2 is not None
    assert path2[0] == start


def test_gradient_field_all_sea_explicit_map():
    # 7x5 map (w=7,h=5) 全て海 (0)
    W, H = 7, 5
    h = HexArray(W, H)
    # place goal near center
    goal = Position(x=3, y=2)
    dist = h.gradient_field(goal)
    # center is zero
    assert dist[2][3] == 0
    # check a few known hex distances: use Position.hex_distance for ground truth
    pairs = [((3,2),(3,2)), ((2,2),(3,2)), ((1,2),(3,2)), ((3,0),(3,2)), ((6,4),(3,2))]
    for (sx, sy), (gx, gy) in pairs:
        p = Position(x=sx, y=sy)
        expected = p.hex_distance(Position(x=gx, y=gy))
        assert dist[sy][sx] == expected


def test_gradient_field_single_land_obstacle():
    # 7x5 map with a single land tile that should be impassable when ignore_land=False
    W, H = 7, 5
    h = HexArray(W, H)
    # put a single land tile between start and goal
    h.set(4,2,1)
    goal = Position(x=5, y=2)
    # when not ignoring land, tiles on the land should be INF/unreachable for the wave origin
    dist = h.gradient_field(goal, ignore_land=False)
    # land tile remains INF (cannot stand on it)
    assert dist[2][4] == INF
    # neighboring sea tile distances reflect shortest hex distance avoiding land origins
    # compare with gradient_field(ignore_land=True) which treats land as sea
    dist_ignore = h.gradient_field(goal, ignore_land=True)
    assert dist_ignore[2][4] == 1  # adjacent when ignoring land
    # ensure goal remains zero in both cases
    assert dist[goal.y][goal.x] == 0
    assert dist_ignore[goal.y][goal.x] == 0


def test_write_svg_to_tmp():
    # create a small map, draw SVG and write to tmp_path
    h = HexArray(5, 5)
    # put some land for visual variety
    h.set(2,1,1)
    h.set(3,2,1)
    svg = h.draw(hex_size=16, show_coords=True)
    out = Path("tmp/map.svg")
    out.write_text(svg, encoding="utf-8")
    assert out.exists()
    assert out.stat().st_size > 0


def test_draw_with_values_sequential():
    # create small map and fill values with sequential integers
    W, H = 6, 4
    h = HexArray(W, H)
    # make a few land tiles for variety
    h.set(2,1,1)
    h.set(3,2,1)
    # prepare sequential values 0..W*H-1
    values = [[y * W + x for x in range(W)] for y in range(H)]
    svg = h.draw(hex_size=18, show_coords=True, values=values)
    out = Path("tmp/map_values.svg")
    out.write_text(svg, encoding="utf-8")
    assert out.exists()
    assert out.stat().st_size > 0
