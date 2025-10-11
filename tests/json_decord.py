import sys,os
import json
if __name__ == '__main__':
    sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

from server.services.ai_llm_base import json_loads

def main():
    samples = [
        '{"key": "va\nlue", "content": "aaaa\nbbbb\ncccc", "number": 123, "list": [1, 2, 3]}',
        '{"thinking":"敵空母のHPが極めて低く、撃沈寸前。空母は安全圏の(4,18)に留まり、航空部隊の帰還を待つ。BSQ1はほぼ戦力外で修復優先。BSQ2は帰還中で修復後に再発艦し、敵空母を確実に撃沈するために攻撃を継続する。敵航空部隊の動向に注意しつつ防御を固める。","action":{"carrier_target":null,"launch_target":null}}{"thinking":"敵空母のHPが非常に低く、撃沈寸前。空母は安全圏の(4,18)に留まり、航空部隊の帰還を待つ。BSQ1はほぼ戦力外で修復優先。BSQ2は帰還中で修復後に再発艦し、敵空母を確実に撃沈するために攻撃を継続する。敵航空部隊の動向に注意しつつ防御を固める。","action":{"carrier_target":null,"launch_target":null}}',
    ]
    for text in samples:
        print("Input:", text)
        try:
            dcorded = json_loads(text)
            print("Decoded:", dcorded)
        except Exception as ex:
            print("Error:", ex)


if __name__ == '__main__':
    main()