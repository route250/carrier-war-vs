import sys, os

if __name__ == '__main__':
    # ai_gemini.py is at server/services; project root is two levels up
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from server.schemas import Position
from server.services.hexmap import HexArray
from server.services.turn import next_step_by_gradient, next_step

"""
テストケース: ユニットの移動が目標に向かって直線的に動くか？
"""
def unit_move_test():
    map = HexArray(10, 10)
    units = []
    start_pos: Position = Position( x=0, y=0 )
    target_pos: Position = Position( x=9, y=9 )
    dist_array = map.gradient_field(target_pos, ignore_land=True)

    steps = [ [0 for col in row] for row in dist_array ]
    current_pos = start_pos
    n = 0
    while current_pos != target_pos:
        next_pos = next_step_by_gradient( map, units, current_pos, target_pos, ignore_land = True, dist_array = dist_array )
        print(f"Step {n}: {current_pos}, Next: {next_pos}")
        if next_pos is None:
            print("No path to target.")
            break
        steps[current_pos.y][current_pos.x] = n
        map.set(current_pos.x,current_pos.y,n+1)
        n += 1
        if next_pos == current_pos:
            print("No further movement possible.")
            break
        current_pos = next_pos

    svg1_path = os.path.join( 'tmp', 'unit_move_test1_dist.svg' )
    svg1_text = map.draw( values=dist_array)
    with open(svg1_path, 'w') as f:
        f.write(svg1_text)
    print(f"Distance field SVG written to {svg1_path}")

    svg2_path = os.path.join( 'tmp', 'unit_move_test1_steps.svg' )
    svg2_text = map.draw( values=steps)
    with open(svg2_path, 'w') as f:
        f.write(svg2_text)
    print(f"Steps SVG written to {svg2_path}")

def unit_move_test2():
    map = HexArray(10, 10)
    units = []
    start_pos: Position = Position( x=0, y=0 )
    target_pos: Position = Position( x=9, y=9 )
    dist_array = map.gradient_field(target_pos, ignore_land=True)

    steps = [ [0 for col in row] for row in dist_array ]
    current_pos = start_pos
    n = 0
    while current_pos != target_pos:
        next_pos = next_step( map, units, start_pos, current_pos, target_pos, ignore_land = True, dist_array = dist_array )
        print(f"Step {n}: {current_pos}, Next: {next_pos}")
        if next_pos is None:
            print("No path to target.")
            break
        steps[current_pos.y][current_pos.x] = n
        map.set( current_pos.x, current_pos.y, n+1)
        n += 1
        if next_pos == current_pos:
            print("No further movement possible.")
            break
        current_pos = next_pos

    svg1_path = os.path.join( 'tmp', 'unit_move_test2_dist.svg' )
    svg1_text = map.draw( values=dist_array)
    with open(svg1_path, 'w') as f:
        f.write(svg1_text)
    print(f"Distance field SVG written to {svg1_path}")

    svg2_path = os.path.join( 'tmp', 'unit_move_test2_steps.svg' )
    svg2_text = map.draw( values=steps)
    with open(svg2_path, 'w') as f:
        f.write(svg2_text)
    print(f"Steps SVG written to {svg2_path}")


if __name__ == '__main__':
    unit_move_test()
    unit_move_test2()