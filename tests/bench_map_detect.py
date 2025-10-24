import sys,os
import json
from time import time,sleep
from typing import Literal
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError


if __name__ == '__main__':
    # server/services配下のai_gemini.pyを読み込むため、プロジェクトルート（二階層上）をインポートパスへ追加
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server.services.hexmap import HexArray
from server.services.match import MatchStore
from server.schemas import MatchCreateRequest, MatchJoinRequest, Position
from server.services.ai_openai import OpenAIConfig, CarrierBotOpenAI
from server.services.ai_anthropic import AnthropicConfig, CarrierBotAnthropic
from server.services.ai_gemini import GeminiConfig, CarrierBotGemini
from server.services.ai_iointelligence import CarrierBotIOIntelligence
from server.services.ai_cpu import Config, CarrierBotMedium
from server.services.ai_llm_base import LLMBase, LLMBaseConfig, LLMTokenUsage, parse_output_to_model
from tests.bench_plot import plot_bench_results


# LLMインターフェース実装ごとの利用可否を事前確認する
def check(llm:LLMBase):
    print(f"Checking LLM {llm.name}...")
    if not llm._ensure_client_ready():
        print(f"Skipping {llm.name} because API key is not set or client cannot be initialized.")
        return False
    return True

# Positionのリストを整形済みテキストへ変換（ログ比較用）
def dump_pos_list(pos_list:list[Position]) -> str:
    return "[" + ",".join( [f"({p.x},{p.y})" for p in sorted(pos_list)] ) + "]"

# 確認項目ごとの正答数を集計し、成功率の可視化に利用
class CheckResult:
    def __init__(self, name:str):
        self.name = name
        self.total = 0
        self.correct = 0
    def add_result(self, result:bool):
        self.total += 1
        if result:
            self.correct += 1

# --- LLM出力構造（output_format用BaseModel） ---

class CellTypeResponse(BaseModel):
    """座標が海か陸かをYes/Noで返す。"""
    answer: Literal["yes", "no"]

    @staticmethod
    def to_json_format() -> str:
        return f"{{ \"answer\": 'yes' | 'no' }}"

class LandCountResponse(BaseModel):
    """隣接する陸タイルの数を返す。"""
    land_count: int

    @staticmethod
    def to_json_format() -> str:
        return f"{{ \"land_count\": integer }}"

class NeighborListResponse(BaseModel):
    """隣接座標の一覧を返す。"""
    neighbors: list[Position]

    @staticmethod
    def to_json_format() -> str:
        return f"{{ \"neighbors\": [ {{ \"x\": integer, \"y\": integer }}, ... ] }}"


def get_cached_result( record_json:dict, query:str ) -> str|None:
    data = record_json.get(query)
    if not isinstance(data, dict):
        return None
    answer_raw = data.get('answer')
    result = data.get('result')
    elapsed = data.get('elapsed')
    cost = data.get('cost')
    if isinstance(answer_raw, str) and isinstance(result, bool) and isinstance(elapsed, (int,float)) and isinstance(cost, dict):
        return answer_raw
    return None

def set_cached_result(record_json:dict, query:str,*,
                    answer_raw:str|None=None,
                    result:bool|None=None, 
                    elapsed:float|None=None, 
                    cost:dict|None=None) -> None:
    if query is None or query=="":
        return
    data = record_json.setdefault(query, {})
    if answer_raw is not None:
        data['answer'] = answer_raw
    if result is not None:
        data['result'] = result
    if elapsed is not None:
        data['elapsed'] = elapsed
    if cost is not None:
        data['cost'] = cost


def save_result( filepath:str, result:dict ):
    try:
        os.makedirs( os.path.dirname(filepath), exist_ok=True )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save result to {filepath}: {e}")

def execute_benchmarks():

    try:
        # server/services配下のai_gemini.pyを読み込むため、プロジェクトルート（二階層上）をインポートパスへ追加
        p = Path(__file__).resolve().parents[2] / "config.env"
        if p.exists():
            load_dotenv(dotenv_path=str(p))
    except Exception:
        # 最小構成のテスト環境ではpython-dotenvが無い場合があるため、読み込み失敗は握りつぶして続行
        pass
    load_dotenv("config.env")
    load_dotenv("../config.env")

    recoard_dir_path = os.path.join('tests', 'bench_map_detect_results')

    # テスト用マップデータ（固定レイアウトでLLMの判断精度を検証）
    hmap = HexArray(5, 5)
    map_data = [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ]
    hmap.set_map(map_data)
    values = [ [f"{x},{y}" for x,val in enumerate(row) ] for y,row in enumerate(map_data) ]
    svg_data = hmap.draw( show_coords=False, values=values )
    map_svg_path = os.path.join( recoard_dir_path, 'map.svg')
    with open(map_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_data)

    print("Map data:")
    print(hmap.to_prompt())

    provider = "openai"
    model = "gpt-4.1-nano"
    model = "gpt-5-nano-L"

    case_list = [
        # gpt-3.5
        ('openai', 'gpt-3.5-turbo'),
        # gpt-4
        ('openai', 'gpt-4-turbo'),
        #＃ ## ## ('openai', 'gpt-4'),
        # gpt-4.1
        ('openai', 'gpt-4.1-nano'),
        ('openai', 'gpt-4.1-mini'),
        ('openai', 'gpt-4.1'),
        # gpt-4o
        ('openai', 'gpt-4o-mini'),
        ('openai', 'gpt-4o'),
        # gpt-5
        ('openai', 'gpt-5-nano'),
        ##('openai', 'gpt-5-nano-L'),
        ##('openai', 'gpt-5-nano-M'),
        ##('openai', 'gpt-5-nano-H'),
        ('openai', 'gpt-5-mini'),
        ##('openai', 'gpt-5-mini-L'),
        ##('openai', 'gpt-5-mini-M'),
        ##('openai', 'gpt-5-mini-H'),
        ('openai', 'gpt-5'),
        ('openai', 'gpt-5-codex'),
        ('openai', 'gpt-5-chat'),

        # anthropic claude
        ('anthropic', 'claude-haiku-3'),
        ('anthropic', 'claude-haiku-3.5'),
        ('anthropic', 'claude-haiku-4.5'),
        ('anthropic', 'claude-sonnet-4'),
        ('anthropic', 'claude-sonnet-4.5'),
        ('anthropic', 'claude-Opus-4.1'),

        # google gemini
        ('gemini', 'gemini-2.0-flash-lite'),
        ('gemini', 'gemini-2.5-flash-lite'),
        ('gemini', 'gemini-2.5-flash'),

        # iointelligence
        ('iointelligence', 'gpt-oss-120b'), 
        ('iointelligence', 'gpt-oss-20b'), 
        ('iointelligence', 'deepseek-r1'), 
        ('iointelligence', 'qwen3-coder-480b-a35b-instruct'), 
        ('iointelligence', 'qwen3-next-80b-a3b-instruct'), 
        ('iointelligence', 'qwen3-235b-a22b-thinking-2507'),
        ('iointelligence', 'qwen2-5-vl-32b'),
        ('iointelligence', 'llama4-17b'), 
        ('iointelligence', 'llama3-3-70b'),
        #('iointelligence', 'llama3-2-90b-vision-instruct'), 
    ]

    # モデル候補ごとにマッチを立ち上げ、3種類の質問テンプレートで精度を測定
    for provider,model in case_list:
        store = MatchStore()
        resp = store.create(MatchCreateRequest(mode="pvp", config=None, display_name="HUMAN"))
        match_id = resp.match_id
        match = store._matches[match_id]

        # LLMボットをB側に参加（スレッドは起動しない）
        if provider == 'openai':
            config = OpenAIConfig(model=model)
            ai_model = CarrierBotOpenAI.get_model_names().get_model(model)
            bot = CarrierBotOpenAI(store=store, match_id=match_id, name=model, config=config)
        elif provider == 'anthropic':
            config = AnthropicConfig(model=model)
            ai_model = CarrierBotAnthropic.get_model_names().get_model(model)
            bot = CarrierBotAnthropic(store=store, match_id=match_id, name=model,config=config)
        elif provider == 'gemini':
            config = GeminiConfig(model=model)
            ai_model = CarrierBotGemini.get_model_names().get_model(model)
            bot = CarrierBotGemini(store=store, match_id=match_id, name=model, config=config)
        elif provider == 'iointelligence':
            ai_model = CarrierBotIOIntelligence.get_model_names().get_model(model)
            bot = CarrierBotIOIntelligence(store=store, match_id=match_id, name=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        # APIキーが未設定などで呼び出せない場合は、以降の検証自体をスキップ
        if not bot._ensure_client_ready():
            print(f"Skipping {provider} LLM bot test because API key is not set or client cannot be initialized.")
            return

        recoard_file_path = os.path.join(recoard_dir_path, f"{model}.json")
        logdict = {}
        try:
            if os.path.exists(recoard_file_path):
                with open(recoard_file_path, "r", encoding="utf-8") as f:
                    logdict = json.load(f)
        except:
            logdict = {}

        logdict['provider'] = provider
        logdict['model'] = model
        logdict['reasoning'] = bot.aimodel.reasoning

        chk1 = CheckResult(name="CellType")
        chk2 = CheckResult(name="Count")
        chk3 = CheckResult(name="Neighbors")
        check_list = [chk1, chk2, chk3]

        st = 0
        # 対角線上の5座標を対象に、セル種別→隣接陸数→隣接リストの順で質問を行う
        for i,pos in enumerate([(0,0),(1,1),(2,2),(3,3),(4,4)]):
            npos = Position(x=pos[0], y=pos[1])
            val = hmap.get(pos[0], pos[1])
            query_value = val if i%2==0 else 1 if val==0 else 0
            query = f"座標{pos}は、{'海' if query_value==0 else '陸'}ですか？ JSON形式で回答し、'answer' フィールドに yes または no を指定して下さい。"
            valid_answer = "yes" if query_value==val else "no"
            print(f"Test case {i+1}: pos:{pos} = {val} ({'海' if val%2==0 else '陸'})")
            print(f"Query: {query} (valid answer: {valid_answer})")
            answer_raw = get_cached_result(logdict, query)
            if answer_raw is None:
                inputs = [
                    {"role": "system", "content": f"You are a carrier commander in a hex-based naval battle game.\n The map layout is as follows:\n{hmap.to_prompt()}\n"},
                    {"role": "user", "content": query}
                ]
                bot.reset_token_usage()
                start_tmp = time()
                answer_raw = bot.LLM(inputs, output_format=CellTypeResponse)
                set_cached_result(logdict, query,
                                  answer_raw=answer_raw,
                                  elapsed=time()-start_tmp, cost=bot._total_token_usage.to_usage_dict(ai_model))
                sleep(5)

            print(f"Answer(raw): {answer_raw}")
            answer_data = parse_output_to_model(answer_raw, output_format=CellTypeResponse)
            if answer_data and answer_data.answer == valid_answer:
                print(f"Parsed: {answer_data.answer}")
                print("Result: Correct\n")
                set_cached_result(logdict, query, result=True )
                chk1.add_result(True)
            else:
                print(f"Parsed: {answer_data}")
                print("Result: Incorrect\n")
                set_cached_result(logdict, query, result=False)
                chk1.add_result(False)

            st+=5

            # 隣接マスを列挙しながら陸（値1）の数を集計
            land_count = 0
            neighbors_pos = []
            for p in hmap.neighbors(npos):
                neighbors_pos.append(p)
                if hmap.get(p.x, p.y) == 1:
                    land_count += 1
            neighbors_pos = sorted(neighbors_pos)
            print(f" Neighbors count: {len(neighbors_pos)}, Land count: {land_count}")
            query = f"座標{pos}に隣接する陸は何マスありますか？ JSON形式で回答し、'land_count' フィールドに整数を入れて下さい。"

            answer_raw = get_cached_result(logdict, query)
            if answer_raw is None:
                inputs = [
                    {"role": "system", "content": f"You are a carrier commander in a hex-based naval battle game.\n The map layout is as follows:\n{hmap.to_prompt()}\n"},
                    {"role": "user", "content": query}
                ]
                print(f"Query: {query} (valid answer: {land_count})")
                bot.reset_token_usage()
                start_tmp = time()
                answer_raw = bot.LLM(inputs, output_format=LandCountResponse)
                set_cached_result(logdict, query,
                                  answer_raw=answer_raw,
                                  elapsed=time()-start_tmp,
                                  cost=bot._total_token_usage.to_usage_dict(ai_model))
                sleep(5)
            print(f"Answer(raw): {answer_raw}")
            answer_data = parse_output_to_model(answer_raw, output_format=LandCountResponse)
            if answer_data and answer_data.land_count == land_count:
                print("Result: Correct\n")
                set_cached_result(logdict, query, result=True )
                chk2.add_result(True)
            else:
                print(f"Parsed: {answer_data}")
                print("Result: Incorrect\n")
                set_cached_result(logdict, query, result=False )
                chk2.add_result(False)

            st+=5

            # 隣接座標の完全リストを文字列比較できる形式で回答させる
            query = f"座標{pos}に隣接するマスの座標を全て教えて下さい。 JSON形式で回答し、'neighbors' フィールドに (x,y) 座標の配列を入れて下さい。"
            valid_answer = dump_pos_list(neighbors_pos)
            answer_raw = get_cached_result(logdict, query)
            if answer_raw is None:
                inputs = [
                    {"role": "system", "content": f"You are a carrier commander in a hex-based naval battle game.\n The map layout is as follows:\n{hmap.to_prompt()}\n"},
                    {"role": "user", "content": query}
                ]
                print(f"Query: {query}")
                bot.reset_token_usage()
                start_tmp = time()
                answer_raw = bot.LLM(inputs, output_format=NeighborListResponse)
                set_cached_result(logdict, query,
                                  answer_raw=answer_raw,
                                  elapsed=time()-start_tmp,
                                  cost=bot._total_token_usage.to_usage_dict(ai_model))
                sleep(5)
            print(f"Answer(raw): {answer_raw}")
            answer_data = parse_output_to_model(answer_raw, output_format=NeighborListResponse)
            answer_decoded = dump_pos_list(answer_data.neighbors) if answer_data else None
            print(f"Valid  : {valid_answer}")
            print(f"Decoded: {answer_decoded}")
            if answer_decoded and answer_decoded == valid_answer:
                print("Result: Correct\n")
                set_cached_result(logdict, query, result=True )
                chk3.add_result(True)
            else:
                print("Result: Incorrect\n")
                set_cached_result(logdict, query, result=False )
                chk3.add_result(False)

            st+=5
        # logdictから、elapsedを集計する
        total_elapsed = 0
        total_prompt_tokens = 0
        total_cache_read_tokens = 0
        total_cache_write_tokens = 0
        total_completion_tokens = 0
        total_reasoning_tokens = 0
        for k,v in logdict.items():
            if isinstance(v, dict) and 'elapsed' in v and 'cost' in v:
                t = v.get('elapsed',0)
                total_elapsed = round( total_elapsed + t, 8 )
                c = v.get('cost')
                if isinstance(c, dict):
                    total_prompt_tokens += c.get('prompt_tokens',0)
                    total_cache_read_tokens += c.get('cache_read_tokens',0)
                    total_cache_write_tokens += c.get('cache_write_tokens',0)
                    total_completion_tokens += c.get('completion_tokens',0)
                    total_reasoning_tokens += c.get('reasoning_tokens',0)

        total_usage = LLMTokenUsage(
            prompt=total_prompt_tokens,
            cache_read=total_cache_read_tokens,
            cache_write=total_cache_write_tokens,
            completion=total_completion_tokens,
            reasoning=total_reasoning_tokens
        )

        print(f"Model: {bot.name}, {bot.aimodel.model}, Reasoning:{bot.aimodel.reasoning}")
        logdict['elapsed'] = total_elapsed

        print("Summary:")
        sum_dict = logdict.setdefault("summary", {})
        chk_total = 0
        chk_correct = 0
        for chk in check_list:
            print(f" {chk.name}: {chk.correct}/{chk.total} correct ({(chk.correct/chk.total*100) if chk.total>0 else 0:.1f}%)")
            d = sum_dict.setdefault(chk.name, {})
            d['correct'] = chk.correct
            d['total'] = chk.total
            chk_total += chk.total
            chk_correct += chk.correct
        print(f" Total: {chk_correct}/{chk_total} correct ({(chk_correct/chk_total*100) if chk_total>0 else 0:.1f}%)")
        d = sum_dict.setdefault("Total", {})
        d['correct'] = chk_correct
        d['total'] = chk_total

        cost_dict = logdict.setdefault("cost", {})
        print(f"Cost: {provider} {model}")
        total_doller = 0.0
        for k,v in total_usage.to_usage_dict(ai_model).items():
            print(f" {k}: {v}")
            cost_dict[k] = v
            if k == 'total_price':
                total_doller = v

        save_result(recoard_file_path, logdict)
        # 為替レートは1ドル150円で計算（実際のレートとは異なる場合があります）
        total_yen = round( total_doller*150, 8)

        print("Done.")
        print(f"Estimated cost: ${total_doller:.4f} (about ¥{total_yen:.3f})")
        print(f"Elapsed time: {total_elapsed:.3f} seconds")
        print()

def main():
    execute_benchmarks()
    plot_bench_results()

if __name__ == '__main__':
    main()
