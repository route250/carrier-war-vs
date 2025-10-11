import json
from typing import overload
from server.schemas import INF, Position

# odd-r の隣接差分（Position.offset_neighbors と一致する集合）
OFF_EVEN: tuple[tuple[int,int], ...] = ((+1, 0), (0, -1), (-1, -1), (-1, 0), (-1, +1), (0, +1))
OFF_ODD:  tuple[tuple[int,int], ...] = ((+1, 0), (+1, -1), (0, -1), (-1, 0), (0, +1), (+1, +1))

def generate_connected_map(map: 'HexArray',*, excludes:list[Position]|None= None, blobs: int = 10, seed: int|None = None) -> None:
    """
    海・陸のblobをランダム配置し、全海タイルが到達可能な地形を生成する。
    hexarray.mを書き換える。
    """
    import random
    r = random.Random(seed)
    W, H = map.W, map.H
    for _attempt in range(60):
        new_map = [[0 for _ in range(W)] for __ in range(H)]
        for _ in range(blobs):
            cx = r.randint(2, max(2, W - 3))
            cy = r.randint(2, max(2, H - 3))
            rad = r.randint(1, 3)
            for dy in range(-rad, rad + 1):
                for dx in range(-rad, rad + 1):
                    if dx * dx + dy * dy <= rad * rad:
                        x = max(0, min(W - 1, cx + dx))
                        y = max(0, min(H - 1, cy + dy))
                        new_map[y][x] = 1
            if excludes:
                for pos in excludes:
                    new_map[pos.y][pos.x] = 0   
        map.set_map(new_map)
        if map.validate_sea_connectivity():
            return

def encode_map( raw_map: list[list[int]] ) -> dict[int, list[tuple[int,int,int]]]:
    """
    ヘックスマップを圧縮して辞書形式で返す。
    キーは、行番号、値はその行の陸地セルの列範囲と値(s,e,value)タプルのリスト。
    """
    encoded: dict[int,list[tuple[int,int,int]]] = {}
    for y, row in enumerate(raw_map):
        ranges = []
        start = -1
        current_value = 0
        for x, cell in enumerate(row):
            if cell != current_value:
                if start >= 0:
                    ranges.append((start, x - 1, current_value))
                if cell != 0:
                    start = x
                    current_value = cell
                else:
                    start = -1
                    current_value = 0
        if current_value != 0 and start >= 0:
            ranges.append((start, len(row) - 1,current_value))
        if ranges:
            encoded[y] = ranges
    return encoded

def unset_encoded_map(encoded: dict[int, list[tuple[int,int,int]]], x:int, y:int) -> None:

    if y not in encoded:
        return
    ranges = encoded[y]
    if not ranges:
        del encoded[y]
        return

    # 0に設定する場合、該当範囲を削除または分割
    for i,(s,e,v) in enumerate(ranges):
        if s <= x <= e:
            if s == e:
                ranges.pop(i)
                if not ranges:
                    del encoded[y]
            elif s == x:
                ranges[i] = (s + 1, e, v)
            elif e == x:
                ranges[i] = (s, e - 1, v)
            else:
                ranges[i] = (s, x - 1, v)
                ranges.insert(i + 1, (x + 1, e, v))
            return
        elif x < s:
            break
    return

def rawmap_to_text( raw_map: list[list[int]] ):
    """ デバッグ用にヘックスマップをテキストで表示する。 odd-rタイプ"""
    for y, row in enumerate(raw_map):
        indent = " " if y % 2 == 1 else ""
        sep = " " if y % 2 == 1 else " "
        line = sep.join(str(cell)[-1] for cell in row)
        yield indent+line

def mearge_encoded_ranges( ranges: list[tuple[int,int,int]], i1:int, i2:int ):
    if 0<=i2<len(ranges)-1:
        s1,e1,v1 = ranges[i2]
        s2,e2,v2 = ranges[i2+1]
        if v1 == v2 and e1 + 1 >= s2:
            # merge
            ranges[i2] = (s1, e2, v1)
            ranges.pop(i2+1)
    if 0<i1:
        s1,e1,v1 = ranges[i1-1]
        s2,e2,v2 = ranges[i1]
        if v1 == v2 and e1 + 1 >= s2:
            # merge
            ranges[i1-1] = (s1, e2, v1)
            ranges.pop(i1)

def set_encoded_map(encoded: dict[int, list[tuple[int,int,int]]], x:int, y:int, value ) -> None:
    """
    encode_mapの結果に対して、指定された位置に値を設定する。
    """
    if value == 0:
        unset_encoded_map(encoded, x, y)
        return

    if y not in encoded:
        # その行がまだ存在しない場合は単純に追加
        encoded[y] = [(x, x, value)]
        return

    ranges = encoded[y]

    # 0以外に設定する場合、範囲を拡張または更新
    if not ranges:
        encoded[y] = [(x, x, value)]
        return
    for i,(s,e,v) in enumerate(ranges):
        if x < s-1:
            ranges.insert(i, (x, x, value))
            mearge_encoded_ranges(ranges, i, -1)
            return
        elif x == s-1:
            if v == value:
                ranges[i] = (x, e, v)
            else:
                ranges.insert(i, (x, x, value))
            mearge_encoded_ranges(ranges, i, -1)
            return
        elif s == x:
            if v != value:
                if s==e:
                    ranges[i] = (s, s, value)
                    mearge_encoded_ranges(ranges, i, i)
                else:
                    ranges[i] = (s+1, e, v)
                    ranges.insert(i, (s, s, value))
                    mearge_encoded_ranges(ranges, i, -1)
            return
        elif s < x < e:
            if v != value:
                ranges[i] = (s, x - 1, v)
                ranges.insert(i + 1, (x, x, value))
                ranges.insert(i + 2, (x + 1, e, v))
            return
        elif x == e:
            if v != value:
                ranges[i] = (s, e - 1, v)
                ranges.insert(i + 1, (e, e, value))
                mearge_encoded_ranges(ranges, -1, i+1)
            return
        elif x == e + 1:
            if v == value:
                ranges[i] = (s, x, v)
                mearge_encoded_ranges(ranges, -1, i)
            else:
                ranges.insert(i + 1, (x, x, value))
                mearge_encoded_ranges(ranges, -1, i+1)
            return
        elif x < s:
            ranges.insert(i, (x, x, value))
            mearge_encoded_ranges(ranges, i, i)
            return
    # どれにも該当しない場合は最後に追加
    ranges.append((x, x, value))

def decode_map(encoded: dict[int, list[tuple[int,int,int]]], width: int, height: int) -> list[list[int]]:
    """
    encode_mapの逆変換。圧縮された辞書形式から2次元配列を復元する。
    """
    raw_map = [[0 for _ in range(width)] for __ in range(height)]
    for y, ranges in encoded.items():
        if 0 <= y < height:
            for s, e, value in ranges:
                if s < 0 or e >= width or s > e:
                    raise ValueError(f"Invalid range in encoded map: row {y}, range ({s},{e})")
                for x in range(s, e + 1):
                    raw_map[y][x] = value
    return raw_map


class HexArray:
    """
    ヘックスマップを2次元配列で表現するクラス。
    各セルは整数値を持ち、0は海、非0は陸を表す。
    """
    def __init__(self, width: int, height: int):
        self.__map = [[0 for _ in range(width)] for __ in range(height)]
        self.__W = width
        self.__H = height

    def set_map(self, values: list[list[int]]):
        if not isinstance(values, list) or not all(isinstance(row, list) for row in values):
            raise ValueError("m must be a 2D list")
        H = len(values)
        W = len(values[0]) if H > 0 else 0
        if any(len(row) != W for row in values):
            raise ValueError("All rows in m must have the same length")
        self.__map = values
        self.__W = W
        self.__H = H

    @property
    def W(self) -> int:
        return self.__W

    @property
    def H(self) -> int:
        return self.__H

    @property
    def shape(self) -> tuple[int, int]:
        return (self.W, self.H)

    def copy_as_list(self) -> list[list[int]]:
        return [row[:] for row in self.__map]

    def get(self, x:int, y:int,) -> int:
        """指定された位置のセルの値を取得する。"""
        if not (0 <= x < self.W and 0 <= y < self.H):
            raise IndexError("Coordinates out of bounds")
        return self.__map[y][x]

    def set(self, x:int, y:int, value:int) -> None:
        if not (0 <= x < self.W and 0 <= y < self.H):
            raise IndexError("Coordinates out of bounds")
        self.__map[y][x] = value

    def __getitem__(self, pos: Position) -> int:
        if not isinstance(pos, Position):
            raise TypeError(f"HexArray indices must be Position, not {type(pos).__name__}")
        if not (0 <= pos.x < self.W and 0 <= pos.y < self.H):
            raise IndexError("Coordinates out of bounds")
        return self.__map[pos.y][pos.x]

    def __setitem__(self, pos:Position, value:int ):
        """指定された位置のセルの値を設定する。"""
        if not isinstance(pos, Position):
            raise TypeError(f"pos must be Position, not {type(pos).__name__}")
        if not (0 <= pos.x < self.W and 0 <= pos.y < self.H):
            raise IndexError("Coordinates out of bounds")
        self.__map[pos.y][pos.x] = value

    def gradient_field(self, goal: Position, ignore_land: bool = False) -> list[list[int]]:
        """
        グラデーション波形の距離フィールドを計算して2次元リストで返す。
        goal: 目標位置
        ignore_land: Trueなら陸地を無視（海のみ通行可にしない）
        stop_range: ゴール近傍R以内の複数セルをソースにする半径

        実装メモ: PydanticのPosition生成/メソッド呼び出しを避け、整数座標のみでBFS実装。
        """
        W = self.W
        H = self.H
        if W == 0 or H == 0:
            return []

        grid = self.__map  # ローカル参照で高速化
        dist: list[list[int]] = [[INF for _ in range(W)] for __ in range(H)]

        def passable_xy(x: int, y: int) -> bool:
            if not (0 <= x < W and 0 <= y < H):
                return False
            if not ignore_land and grid[y][x] != 0:
                return False
            return True

        gx, gy = goal.x, goal.y

        from collections import deque
        q: 'deque[tuple[int,int]]' = deque()

        # 最頻パス: stop_range==0 は単一点ソースで十分
        if not passable_xy(gx, gy):
            return dist

        dist[gy][gx] = 0
        q.append((gx, gy))
        # BFS
        while q:
            x, y = q.popleft()
            cd = dist[y][x]
            offs = OFF_ODD if (y & 1) else OFF_EVEN
            nd = cd + 1
            for dx, dy in offs:
                nx = x + dx
                ny = y + dy
                if not passable_xy(nx, ny):
                    continue
                if dist[ny][nx] > nd:
                    dist[ny][nx] = nd
                    q.append((nx, ny))

        return dist

    def adjust_to_sea(self, pos: Position ) -> Position|None:
        """指定地点に最も近い海タイルを探して返す"""
        if self[pos] == 0:
            return pos # 指定地点は海タイル
        # 距離フィールドを計算
        dist = self.gradient_field(pos, ignore_land=True)
        sea_pos = None
        min_d = INF
        for y in range(self.H):
            for x in range(self.W):
                if self.__map[y][x] == 0 and dist[y][x] < min_d:
                    min_d = dist[y][x]
                    sea_pos = Position(x=x, y=y)
        return sea_pos

    def validate_sea_connectivity(self) -> bool:
        """
        全ての海タイルが互いに到達可能か検証する。戻り値: (ok, sea_total, sea_reached)
        """
        # 海の任意の一点を探す
        sea_pos = None
        for y in range(self.H):
            for x in range(self.W):
                if self.__map[y][x] == 0:
                    sea_pos = Position(x=x, y=y)
        if sea_pos is None:
            # 海が存在しない場合はダメ
            return False
        # 距離フィールドを計算
        dist = self.gradient_field(sea_pos, ignore_land=False)
        # 全ての海タイルが到達可能か確認
        for y in range(self.H):
            for x in range(self.W):
                if self.__map[y][x] == 0 and dist[y][x] == INF:
                    return False
        return True

    def path_by_gradient(self, start: Position, goal: Position, ignore_land: bool = False, max_steps: int = 5000) -> list:
        """
        グラデーション波形距離フィールドを使ってstartからgoalまでのパスを復元する。
        neighbors_by_gradientで進行方向を選択。
        Positionのみで処理し、範囲外はneighbors_by_gradientで除外済み前提。
        """
        if self.W == 0 or self.H == 0:
            return [start]
        dist_array = self.gradient_field(goal, ignore_land=ignore_land)
        pos = start
        path = [pos]
        steps = 0
        while steps < max_steps:
            dcur = dist_array[pos.y][pos.x]
            if dcur == 0 or dcur >= INF:
                break
            nbrs = self.neighbors_by_gradient(pos, goal, ignore_land=ignore_land, dist_array=dist_array)
            if not nbrs:
                break
            next_pos = nbrs[0]
            path.append(next_pos)
            pos = next_pos
            steps += 1
        return path


    def adjust_target(self, origin_pos: Position, target_pos: Position, ignore_island:bool ) -> Position|None:
        """オーダーの目標がエラーの場合、目標を修正する。移動できない場合はNoneを返す。"""
        if origin_pos == target_pos:
            return None
        # target_posがマップ範囲外なら、origin_pos→target_posの方向を維持したまま
        # マップ矩形の最近傍の境界との交点に射影する
        if not (0 <= target_pos.x < self.W and 0 <= target_pos.y < self.H):
            pos = self.project_to_map_edge(origin_pos, target_pos)
            if pos is None:
                return None
            target_pos = pos
        if not ignore_island and self[target_pos] != 0:
            # 陸地を除外するけど、目的地が陸地
            pos = self.adjust_to_sea(target_pos)
            if pos is None or self[pos] != 0:
                return None
            target_pos = pos
        return target_pos


    def neighbors(self, pos: Position):
        for npos in pos.offset_neighbors():
            if 0 <= npos.x < self.W and 0 <= npos.y < self.H:
                yield npos

    def neighbors_by_gradient(self, start: Position, goal: Position, ignore_land: bool = False, dist_array:list[list[int]]|None = None) -> list[Position]:
        """
        startの周囲6方向のPositionを、goalへの距離が近い順に並べて返す。
        距離が同じ場合はgoal方向との角度差が小さい順で優先。
        """
        if start == goal:
            return []
        dist = dist_array if dist_array else self.gradient_field(goal, ignore_land=ignore_land)
        base_vector = start.normalized_vector(goal)
        neighbors = []
        for npos in self.neighbors(start):
            d = dist[npos.y][npos.x]
            if 0<= d < INF:
                vec = start.normalized_vector(npos)
                dot = base_vector[0] * vec[0] + base_vector[1] * vec[1]
                neighbors.append((d, -dot, npos))
        neighbors.sort()
        return [npos for _, _, npos in neighbors]

    def neighbors_by_distance(self, start: Position, current: Position, goal: Position, ignore_land: bool = False, dist_array:list[list[int]]|None = None) -> list[Position]:
        """
        startの周囲6方向のPositionを、goalへの距離が近い順に並べて返す。
        距離が同じ場合はコースとの距離で優先。
        """
        sx, sy = start.center_xy()
        ex, ey = goal.center_xy()
        dx = ex - sx
        dy = ey - sy
        denom = (dx ** 2 + dy ** 2) ** 0.5
        if abs(dx)<1e-9 and abs(dy)<1e-9 or denom<1e-9:
            return []
        dist = dist_array if dist_array else self.gradient_field(goal, ignore_land=ignore_land)
        neighbors = []
        for npos in self.neighbors(current):
            d = dist[npos.y][npos.x]
            if 0<= d < INF:
                x,y = npos.center_xy()
                # 直線 (sx, sy)-(ex, ey) と点 (x, y) の距離を計算
                num = abs(dy * x - dx * y + ex * sy - ey * sx)
                dist_to_line = num / denom
                neighbors.append((d, dist_to_line, npos))
        neighbors.sort()
        return [npos for _, _, npos in neighbors]

    def distance(self, start: Position, goal: Position, ignore_land: bool = False) -> int:
        """
        起点と目標をPositionで受け取り、距離をintで返す。
        ignore_land=Trueなら陸地を無視（現状は未実装、必要なら地形判定を追加）
        """
        if not isinstance(start, Position) or not isinstance(goal, Position):
            raise TypeError("start/goal must be Position")
        return start.hex_distance(goal)

    def project_to_map_edge(self, origin_pos: Position, target_pos: Position) -> Position|None:
        ox, oy = origin_pos.x, origin_pos.y
        dx = target_pos.x - ox
        dy = target_pos.y - oy
        if dx == 0 and dy == 0:
            return None
        # t値（origin + t*(dx,dy)）で最初に到達する境界を求める
        INF = 10**18
        tx = INF
        ty = INF
        if dx > 0:
            tx = (self.W - 1 - ox) / dx
        elif dx < 0:
            tx = (0 - ox) / dx
        if dy > 0:
            ty = (self.H - 1 - oy) / dy
        elif dy < 0:
            ty = (0 - oy) / dy
        t = tx if tx < ty else ty
        # 交点（片方の座標はちょうど境界になる）
        if t == tx:
            nx = self.W - 1 if dx > 0 else 0
            nyf = oy + dy * t
            ny = int(round(nyf))
            if ny < 0:
                ny = 0
            elif ny >= self.H:
                ny = self.H - 1
            return Position(x=nx, y=ny)
        else:
            ny = self.H - 1 if dy > 0 else 0
            nxf = ox + dx * t
            nx = int(round(nxf))
            if nx < 0:
                nx = 0
            elif nx >= self.W:
                nx = self.W - 1
            return Position(x=nx, y=ny)

    def path_by_AStar(self, start: Position, goal: Position, ignore_land: bool = False, stop_range: int = 0, max_expand: int = 4000) -> list|None:
        """
        できたら使わないようにする!
        A*によるパス探索。地形のみ考慮。ignore_land=Trueなら陸地を無視。
        stop_range: ゴール判定範囲
        """
        W = self.W
        H = self.H
        if W == 0 or H == 0:
            return None
        def in_bounds(x, y):
            return 0 <= x < W and 0 <= y < H
        def passable(pos: Position):
            if not in_bounds(pos.x, pos.y):
                return False
            if not ignore_land and self.__map[pos.y][pos.x] != 0:
                return False
            return True
        if not passable(start):
            return None
        if start.hex_distance(goal) <= max(0, stop_range):
            return [start]
        import heapq
        open_heap = []
        heapq.heappush(open_heap, (0 + start.hex_distance(goal), 0, start))
        came_from = {start: None}
        g_score = {start: 0}
        closed = set()
        expands = 0
        while open_heap and expands < max_expand:
            f, g, pos = heapq.heappop(open_heap)
            if pos in closed:
                continue
            closed.add(pos)
            expands += 1
            if pos.hex_distance(goal) <= max(0, stop_range):
                path = [pos]
                cur = pos
                while cur and came_from[cur] is not None:
                    cur = came_from[cur]
                    if cur:
                        path.append(cur)
                path.reverse()
                return path
            for npos in pos.offset_neighbors():
                if not passable(npos):
                    continue
                tentative = g + 1
                if tentative < g_score.get(npos, 1e9):
                    g_score[npos] = tentative
                    came_from[npos] = pos
                    h = npos.hex_distance(goal)
                    heapq.heappush(open_heap, (tentative + h, tentative, npos))
        return None

    def dump_row_itr(self):
        for line in rawmap_to_text(self.__map):
            yield line

    def dump_to_str(self) -> str:
        lines = []
        fmt = "{"+f"{len(str(self.H-1))}d"+"}"
        for y, row in enumerate(self.dump_row_itr()):
            head = ("    "+str(y))[-2:]
            lines.append(f"{head}: {row}")
        return "\n".join(lines)

    def to_prompt_txt(self) -> str:
        lines = []
        lines.append(f"hexamap: width:{self.W} height:{self.H}, sea=0 land=1")
        for y, row in enumerate(self.dump_row_itr()):
            head = ("    "+str(y))[-2:]
            lines.append(f"{head}: {row}")
        return "\n".join(lines)

    def to_prompt(self) -> str:
        lines = []
        lines.append("```")
        lines.append("- gridType: hex, orientation: pointy-top, offset: odd-r, indexBase: 0")
        lines.append(f"- width: {self.W}, height: {self.H}")
        lines.append("- neighbors are defined by the following deltas:")
        lines.append("  even(y%2==0): (-1,0),(+1,0),(-1,-1),(0,-1),(-1,+1),(0,+1)")
        lines.append("  odd (y%2==1): (-1,0),(+1,0),(0,-1),(+1,-1),(0,+1),(+1,+1)")
        lines.append("- exclude out-of-board positions.")
        lines.append("- legend: 0=sea, non0=land")
        lines.append("- tiles:")
        for y, row in enumerate(self.dump_row_itr()):
            head = "    "
            lines.append(f"{head}{row}")
        lines.append("```")
        return "\n".join(lines)

    def dump(self):
        """キャラクタベースのヘックスマップをプリントする。
        偶数/奇数行をインデントして六角形グリッドの視覚的なズレを表現します。
        0 を海 (.)、非0 を陸 (#) として表示します。
        """
        yy = "  "
        for x, col in enumerate(self.__map[0]):
            yy += f" {x:2d}"
        print(yy)
        for y, row in enumerate(self.__map):
            # 奇数行を少しインデント（見やすさ向上）
            yy = f"{y:2d}: "
            prefix = "  " if y % 2 == 1 else ""
            chars = []
            for cell in row:
                chars.append(f"{cell}")
            print(yy + prefix + "  ".join(chars))

    def draw(self, *,
            hex_size: int = 20, show_coords: bool = True, values: list[list]|None = None,
            sea_color: str = "#9dd3ff", land_color: str = "#F9EB9C", stroke_color: str = "#444444",
        ) -> str:
        """
        SVG 文字列でマップを描画して返す。
        hex_size: 六角形の半径(px)
        sea_color, land_color, stroke_color: CSS カラー文字列
        show_coords: 各セル中央に座標を表示するか

        レイアウトは pointy-top のオフセット行 (odd-r) を採用。
        static/main.js と同一の座標系/計算式（odd-r offset + axial ベース）に合わせる。
        """
        import math

        # helper: compute polygon points for a hex at pixel center (cx, cy)
        def hex_corners(cx, cy, r):
            pts = []
            for i in range(6):
                angle = math.pi / 180 * (60 * i - 30)  # pointy top
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                pts.append((x, y))
            return pts

        rows = self.H
        cols = self.W
        r = float(hex_size)
        SQRT3 = math.sqrt(3.0)
        ORIGIN_X = r
        ORIGIN_Y = r
        svg_width = SQRT3 * r * (cols + 0.5)
        svg_height = 1.5 * r * (rows - 1) + 2 * r

        parts = []
        parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width:.0f}" height="{svg_height:.0f}" viewBox="0 0 {svg_width:.2f} {svg_height:.2f}">')
        parts.append(f'<rect width="100%" height="100%" fill="white"/>')

        # label font scaled to hex size
        label_font = max(6, int(r / 3))

        for y in range(rows):
            for x in range(cols):
                # compute center for odd-r pointy-top offset (client parity/式に完全一致)
                cx = r * (SQRT3 * (x + 0.5 * (y & 1))) + ORIGIN_X
                cy = r * (1.5 * y) + ORIGIN_Y

                pts = hex_corners(cx, cy, r)
                pts_str = " ".join(f'{px:.2f},{py:.2f}' for px, py in pts)
                cell = self.__map[y][x]
                color = sea_color if cell == 0 else land_color
                parts.append(f'<polygon points="{pts_str}" fill="{color}" stroke="{stroke_color}" stroke-width="1"/>')
                # If values provided, validate shape and render value inside hex (centered).
                if values is not None:
                    try:
                        v = str(values[y][x])
                    except Exception:
                        raise ValueError("values must be a 2D list with same shape as self.map")
                    # render value at hex center
                    value_y = cy + (r*0.2)
                    parts.append(
                        f'<text x="{cx:.2f}" y="{value_y:.2f}" font-size="{label_font}" text-anchor="middle" fill="#111" font-family="monospace" pointer-events="none" dominant-baseline="middle">{v}</text>'
                    )
                if show_coords:
                    # place coordinate label slightly below the top edge of the hex
                    # top edge y is approximately cy - r
                    # move it down a bit so it sits just under the edge (0.55..0.7 of r)
                    top_y = cy - r
                    label_y = top_y + (r * 0.6)
                    # small font and vertically centered via dominant-baseline
                    parts.append(
                        f'<text x="{cx:.2f}" y="{label_y:.2f}" font-size="{label_font}" text-anchor="middle" fill="#111" font-family="monospace" pointer-events="none" dominant-baseline="middle">{x},{y}</text>'
                    )

        parts.append('</svg>')
        return "\n".join(parts)
