import sys,os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.services.hexmap import generate_connected_map, encode_map, decode_map, set_encoded_map, rawmap_to_text, HexArray

def test_map_encoder():
    map_data = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 2, 2, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    for line in rawmap_to_text(map_data):
        print(line)

    W = len(map_data[0])
    H = len(map_data)
    encoded = encode_map(map_data)
    print()
    print(f"Encoded map: {encoded}")
    decoded = decode_map(encoded,W,H)
    assert decoded == map_data

    map2_data = {}
    for r, row in enumerate(map_data):
        for c, val in enumerate(row):
            if val != 0 and c % 2 == 0:
                set_encoded_map(map2_data, c, r, val)
        for c, val in enumerate(row):
            if val != 0 and c % 2 == 1:
                set_encoded_map(map2_data, c, r, val)
    print()
    print(f"Encoded map: {map2_data}")
    decoded = decode_map(map2_data,W,H)
    assert decoded == map_data
    print()

def test_hexmap_encode():
    hmap = HexArray(50,50)
    generate_connected_map(hmap)
    raw_map = hmap.copy_as_list()
    encoded = encode_map(raw_map)
    print(f"Encoded map: {encoded}")
    decoded = decode_map(encoded,hmap.W,hmap.H)
    assert decoded == raw_map

if __name__ == "__main__":
    test_map_encoder()
    test_hexmap_encode()
    print("All tests passed.")