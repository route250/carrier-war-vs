"""LLMボット同士の対戦実行を手動で行うための簡易ベンチツール。"""

import sys, os
import time
import csv

from pandas import Series

if __name__ == '__main__':
    # ai_gemini.py is at server/services; project root is two levels up
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import json
from typing import Any
from html import escape
from pathlib import Path
from dotenv import load_dotenv

try:
    import pandas as pd
except ImportError:  # pragma: no cover - 実行環境に未導入の場合
    pd = None

try:
    import matplotlib
    matplotlib.use('Agg')  # ヘッドレス環境でもSVG出力可能にする
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None

from server.services.ai_base import AIStat
from server.services.match import Match, MatchStore
from server.schemas import MatchCreateRequest, Config
from server.services.ai_openai import OpenAIConfig, CarrierBotOpenAI
from server.services.ai_anthropic import AnthropicConfig, CarrierBotAnthropic
from server.services.ai_gemini import GeminiConfig, CarrierBotGemini
from server.services.ai_iointelligence import CarrierBotIOIntelligence
from server.services.ai_cpu import Config, CarrierBotMedium
from server.services.ai_llm_base import LLMBase, LLMBaseConfig
from tests.plotlib import write_table_svg, write_scatter_svg, resolve_model_category, COLOR_MAP


OUTPUT_DIR = os.path.join('tests', 'bench_llm_competition_results')


def _build_output_path(base_name: str, suffix: str | None, extension: str) -> str:
    suffix_part = f"_{suffix}" if suffix else ""
    return os.path.join(OUTPUT_DIR, f"{base_name}{suffix_part}.{extension}")

def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def write_csv(path: str, headers: list[str], rows: list[list[Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, 'w', newline='', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerow(headers)
        writer.writerows(rows)

def to_competition_id(c1:Config, c2:Config):
    if not c1.llm_model or not c2.llm_model:
        raise ValueError("LLM model not specified")
    return '_vs_'.join(sorted([ c1.llm_model.lower(), c2.llm_model.lower()]))

def to_competition_dir( competition_id:str, nn:int):
    return os.path.join(OUTPUT_DIR, competition_id, f"{nn}")

def to_log_path( competition_id:str, nn:int, model:str):
    return os.path.join(OUTPUT_DIR, competition_id, f"{nn}", f"{model}_log.json")

def to_result_path( competition_id:str, nn:int):
    return os.path.join(OUTPUT_DIR, competition_id, f"{nn}", f"result.json")

def get_result_paths():
    """保存されている対戦結果のJSONファイルパスを全部取得する。"""
    results = []
    if not os.path.exists(OUTPUT_DIR):
        return results
    for competition_id in os.listdir(OUTPUT_DIR):
        competition_dir = os.path.join(OUTPUT_DIR, competition_id)
        if not os.path.isdir(competition_dir):
            continue
        if competition_id.startswith("gpt-5-m_vs") or competition_id.endswith("vs_gpt-5-m"):
            # gpt-5-Mは高価なのでランキング集計から除外する
            continue
        for nn in os.listdir(competition_dir):
            try:
                int(nn)
            except ValueError:
                continue
            result_path = to_result_path(competition_id, int(nn))
            if os.path.exists(result_path):
                results.append(result_path)
    return results

def get_ranking_csv_path(suffix: str | None = None):
    """
    ランキング表のCSV保存先パスを取得する。
    モデル名、勝った回数、勝率(勝った回数/対戦回数)の3列で、勝った回数で降順ソートしたものを保存する。
    """
    return _build_output_path("ranking", suffix, "csv")

def get_ranking_svg_path(suffix: str | None = None):
    """ランキング表のSVG保存先パスを取得する。"""
    return _build_output_path("ranking", suffix, "svg")

def get_cross_table_csv_path(suffix: str | None = None):
    """
    対戦結果表のCSV保存先パスを取得する。
    モデル名を行と列に並べて、各セルに勝敗結果を入れたものを保存する。
    """
    return _build_output_path("cross_table", suffix, "csv")

def get_cross_table_svg_path(suffix: str | None = None):
    """対戦結果表のSVG保存先パスを取得する。"""
    return _build_output_path("cross_table", suffix, "svg")

def get_cost_csv_path():
    """
    コスト表のSVG保存先パスを取得する。
    モデル名、価格総計、一試合あたりの価格、トークン総計、一試合当たりのトークン数、勝った回数、試合回数　を列にする
    """
    return os.path.join(OUTPUT_DIR, f"cost.csv")

def get_cost_table_svg_path():
    """
    コスト表のSVG保存先パスを取得する。
    画像にするときは、一試合あたりの価格の高い順にソートし、モデル名、一試合あたりの価格、一試合あたりのトークン数の3列を画像にする
    """
    return os.path.join(OUTPUT_DIR, f"cost_table.svg")

def get_cost_performance_svg_path():
    """
    コストパフォーマンス図のSVG保存先パスを取得する。
    縦軸に、一試合あたりの価格、横軸に勝った回数で、モデル名をプロットする
    """
    return os.path.join(OUTPUT_DIR, f"cost_performance.svg")


class LLMCompetition:
    """LLMボット（OpenAI/Anthropic/Gemini/CPU）の挙動確認を行う補助テスト。"""

    def __init__(self, c1:Config, c2:Config, nn:int=1):
        if not c1.llm_model or not c2.llm_model:
            raise ValueError("LLM model not specified")
        self.c1 = c1
        self.c2 = c2
        self.competition_id=to_competition_id(c1,c2)
        self.log_c1 = to_log_path(self.competition_id, nn, c1.llm_model)
        self.log_c2 = to_log_path(self.competition_id, nn, c2.llm_model)
        self.log_result = to_result_path(self.competition_id, nn)
        self.bot1 = None
        self.bot2 = None

        # If tests or imports run this module directly, ensure project-local `config.env` is loaded so
        # environment variables like GEMINI_API_KEY are available during import-time.
        try:
            # ai_gemini.py is at server/services; project root is two levels up
            p = Path(__file__).resolve().parents[2] / "config.env"
            if p.exists():
                load_dotenv(dotenv_path=str(p))
        except Exception:
            # dotenv may be unavailable in some minimal test environments; ignore if not present
            pass
        load_dotenv("config.env")
        load_dotenv("../config.env")
        self.store: MatchStore = MatchStore()
        self.match: Match|None = None
        self.threads: list[threading.Thread] = []

    def exists(self):
        return os.path.exists(self.log_c1) and os.path.exists(self.log_c2) and os.path.exists(self.log_result)

    def _create_bot(self, provider, model ):
        """指定したプロバイダとモデルでLLMボットを初期化してマッチに登録する。"""
        if self.store is None or self.match is None:
            raise ValueError("store or match is None")
        # LLMボットをB側に参加（スレッドは起動しない）
        if provider == 'openai':
            config = OpenAIConfig(model=model)
            bot = CarrierBotOpenAI(store=self.store, match_id=self.match.match_id, name=model, config=config)
        elif provider == 'anthropic':
            config = AnthropicConfig(model=model)
            bot = CarrierBotAnthropic(store=self.store, match_id=self.match.match_id, name=model,config=config)
        elif provider == 'gemini':
            config = GeminiConfig(model=model)
            bot = CarrierBotGemini(store=self.store, match_id=self.match.match_id, name=model, config=config)
        elif provider == 'iointelligence':
            #ai_model = CarrierBotIOIntelligence.get_model_names().get_model(model)
            config = LLMBaseConfig(model=model,max_input_tokens=4000)
            bot = CarrierBotIOIntelligence(store=self.store, match_id=self.match.match_id, name=model)
        else:
            bot = CarrierBotMedium(store=self.store, match_id=self.match.match_id, config=None)
            # raise ValueError(f"Unknown provider: {provider}")
        
        if not bot._ensure_client_ready():
            raise ValueError(f"Skipping {provider} LLM bot test because API key is not set or client cannot be initialized.")

        self.match.ai_threads.append(bot)
        t = threading.Thread(target=bot.run, daemon=True)

        self.threads.append(t)
        return bot        

    def eval(self, overwrite: bool=False) -> bool:
        """A/B両軍のボットを生成して同一マッチへ接続する。"""
        if not self.store:
            raise ValueError("store is None")
        # PvPでマッチ作成（A側に人間スロット作成）
        resp = self.store.create(MatchCreateRequest(mode="eve", config=None, display_name="LLM"))
        match_id = resp.match_id
        self.match = self.store._matches[match_id]
        if self.match is None:
            raise ValueError("match is None")

        self.bot1 = self._create_bot( self.c1.provider, self.c1.llm_model )
        if self.bot1 is None or self.bot1.aimodel.name != self.c1.llm_model:
            raise ValueError("bot1 is None")
        if isinstance(self.bot1, CarrierBotIOIntelligence) and os.getenv("IOINTELLIGENCE_API_KEY_1"):
            self.bot1.set_api_key(os.getenv("IOINTELLIGENCE_API_KEY_1"))
 
        self.bot2 = self._create_bot( self.c2.provider, self.c2.llm_model )
        if self.bot2 is None or self.bot2.aimodel.name != self.c2.llm_model:
            raise ValueError("bot2 is None")
        if isinstance(self.bot2, CarrierBotIOIntelligence) and os.getenv("IOINTELLIGENCE_API_KEY_2"):
            self.bot2.set_api_key(os.getenv("IOINTELLIGENCE_API_KEY_2"))

        if not overwrite and self.exists():
            print(f"Skipping existing competition {self.competition_id} {self.bot1.aimodel.name} vs {self.bot2.aimodel.name}")
            return False
        
        # ログ保存先ディレクトリを作成
        os.makedirs( os.path.dirname(self.log_result), exist_ok=True)
        print(f"start {self.competition_id}{self.bot1.aimodel.name} vs {self.bot2.aimodel.name}")
        # メインのベンチシナリオ。2体のボットを起動し停止まで待機する。
        side1 = ""
        side2 = ""
        try:
            for t in self.threads:
                # 各ボットの非同期実行を開始。起動直後のAPI呼び出しが競合しないよう短時間ウェイトを入れる。
                t.start()
                time.sleep(2)

            side1 = self.bot1.side or ""
            if not side1:
                raise ValueError("bot1.side is None")

            side2 = self.bot2.side or ""
            if not side2:
                raise ValueError("bot2.side is None")

            while (self.bot1.stat != AIStat.STOPPED and self.bot1.stat != AIStat.ERROR) or (self.bot2.stat != AIStat.STOPPED and self.bot2.stat != AIStat.ERROR):
                # スレッド終了をポーリング監視。LLM呼び出し中はSTATがRUNNINGのままとなる。
                time.sleep(1)
        except Exception as e:
            print(f"Error occurred: {self.competition_id} {e}")
        finally:
            self.bot1.stop()
            self.bot2.stop()
            for t in self.threads:
                if t.is_alive():
                    # 強制終了を避けるため、短時間だけ join してリソースを解放する。
                    t.join(timeout=1)
        print(f"end {self.competition_id} {self.match.map.result}")

        if self.bot1.stat == AIStat.ERROR or self.bot2.stat == AIStat.ERROR:
            print(f"Error: {self.competition_id} bot1.stat={self.bot1.stat}, bot2.stat={self.bot2.stat}")
            return False

        final_status = self.match.build_state_payload()
        self.bot1.save_history(self.log_c1)
        self.bot2.save_history(self.log_c2)
        # 対戦結果をJSONで保存
        result:dict[str,Any] = {
            "competition_id": self.competition_id,
            'status': final_status.model_dump(),
            'winner_side': self.match.map.result
        }

        if self.match.map.result == side1:
            result['winner_model'] = self.bot1.aimodel.name    
        elif self.match.map.result == side2:
            result['winner_model'] = self.bot2.aimodel.name
        else:
            result['winner_model'] = ''

        result[side1] = {
            "provider": self.c1.provider,
            "model": self.bot1.aimodel.name,
            "model_id": self.bot1.aimodel.model,
            "usage": self.bot1.to_usage_dict(),
            "activity": self.match.get_activity_dict(side1)
        }
        result[side2] = {
            "provider": self.c2.provider,
            "model": self.bot2.aimodel.name,
            "model_id": self.bot2.aimodel.model,
            "usage": self.bot2.to_usage_dict(),
            "activity": self.match.get_activity_dict(side2)
        }

        with open(self.log_result, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        return True

def results():
    """対戦記録を全部読んで、対戦結果表をつくる"""
    if pd is None:
        raise RuntimeError("pandas がインストールされていません。`pip install pandas` を実行してください。")

    paths = sorted(get_result_paths())
    if not paths:
        print("対戦結果が存在しません。先にベンチを実行してください。")
        return

    match_records: list[dict[str, Any]] = []
    usage_records: list[dict[str, Any]] = []
    cross_records: list[dict[str, Any]] = []
    models: set[str] = set()

    for path in paths:
        with open(path, 'r', encoding='utf-8') as fp:
            match_result = json.load(fp)

        path_obj = Path(path)
        match_no = path_obj.parent.name
        competition_id = path_obj.parent.parent.name

        a_side = match_result['A']
        b_side = match_result['B']
        winner_side = match_result.get('winner_side') or match_result.get('winner')
        winner_model = match_result.get('winner_model')

        if winner_side not in ('A', 'B'):
            winner_side = None
        winner_side_label = winner_side if winner_side else ("draw" if not winner_model else "unknown")

        a_model = a_side['model']
        b_model = b_side['model']
        models.update([a_model, b_model])

        a_usage = a_side.get('usage', {}) or {}
        b_usage = b_side.get('usage', {}) or {}

        a_price = float(a_usage.get('total_price') or 0.0)
        b_price = float(b_usage.get('total_price') or 0.0)
        a_tokens = int(a_usage.get('total_tokens') or 0)
        b_tokens = int(b_usage.get('total_tokens') or 0)

        match_records.append(
            {
                "competition_id": competition_id,
                "match_no": match_no,
                "winner": winner_side or "draw",
                "model_a": a_model,
                "model_b": b_model,
                "winner_side": winner_side_label,
                "winner_model": winner_model or "",
                "model_a_total_price": a_price,
                "model_b_total_price": b_price,
                "model_a_total_tokens": a_tokens,
                "model_b_total_tokens": b_tokens,
            }
        )

        usage_records.extend(
            [
                {
                    "model": a_model,
                    "provider": a_side.get('provider'),
                    "side": "A",
                    "competition_id": competition_id,
                    "match_no": match_no,
                    "opponent": b_model,
                    "win": 1 if winner_side == 'A' else 0,
                    "draw": 1 if winner_side is None else 0,
                    "loss": 1 if winner_side == 'B' else 0,
                    "total_price": a_price,
                    "total_tokens": a_tokens,
                },
                {
                    "model": b_model,
                    "provider": b_side.get('provider'),
                    "side": "B",
                    "competition_id": competition_id,
                    "match_no": match_no,
                    "opponent": a_model,
                    "win": 1 if winner_side == 'B' else 0,
                    "draw": 1 if winner_side is None else 0,
                    "loss": 1 if winner_side == 'A' else 0,
                    "total_price": b_price,
                    "total_tokens": b_tokens,
                },
            ]
        )

        if a_model != b_model:
            cross_records.extend(
                [
                    {
                        "row_model": a_model,
                        "col_model": b_model,
                        "wins": 1 if winner_side == 'A' else 0,
                        "losses": 1 if winner_side == 'B' else 0,
                        "draws": 1 if winner_side is None else 0,
                    },
                    {
                        "row_model": b_model,
                        "col_model": a_model,
                        "wins": 1 if winner_side == 'B' else 0,
                        "losses": 1 if winner_side == 'A' else 0,
                        "draws": 1 if winner_side is None else 0,
                    },
                ]
            )

        if winner_side == 'A':
            summary = f"{a_model} (win) vs {b_model} (lose)"
        elif winner_side == 'B':
            summary = f"{a_model} (lose) vs {b_model} (win)"
        else:
            summary = f"{a_model} (draw) vs {b_model} (draw)"
        print(summary)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- pandas DataFrame 化 ---
    matches_df = pd.DataFrame(match_records)
    usage_df = pd.DataFrame(usage_records)
    cross_df = pd.DataFrame(cross_records)

    matches_df = matches_df.sort_values(["competition_id", "match_no", "model_a", "model_b"])
    matches_df.to_csv(os.path.join(OUTPUT_DIR, "match_results.csv"), index=False)

    # --- ランキング集計 ---
    ranking_all_df = (
        usage_df.groupby("model")
        .agg(
            provider=("provider", "first"),
            wins=("win", "sum"),
            draws=("draw", "sum"),
            losses=("loss", "sum"),
            matches=("win", "count"),
            total_price=("total_price", "sum"),
            total_tokens=("total_tokens", "sum"),
        )
        .reset_index()
    )
    ranking_all_df["win_rate"] = ranking_all_df.apply(
        lambda r: (r["wins"] / r["matches"]) if r["matches"] else 0.0,
        axis=1,
    )
    ranking_all_df["price/match"] = ranking_all_df.apply(
        lambda r: (float(r["total_price"]) / r["matches"]) if r["matches"] else 0.0,
        axis=1,
    )
    ranking_all_df["tokens/match"] = ranking_all_df.apply(
        lambda r: (float(r["total_tokens"]) / r["matches"]) if r["matches"] else 0.0,
        axis=1,
    )
    ranking_all_df = ranking_all_df.sort_values(["wins", "win_rate", "model"], ascending=[False, False, True])

    model_providers = (
        usage_df.groupby("model")["provider"].first().to_dict() if not usage_df.empty else {}
    )
    ranking_non_io_df = ranking_all_df[ranking_all_df["provider"] != "iointelligence"].copy()
    ranking_io_df = ranking_all_df[ranking_all_df["provider"] == "iointelligence"].copy()

    ranking_headers = [
        "model",
        "wins",
        "matches",
        "win_rate",
        "price/match",
        "tokens/match",
    ]

    def export_ranking(df, suffix: str | None, title: str) -> None:
        df.to_csv(get_ranking_csv_path(suffix), index=False)
        ranking_rows = [
            [
                row["model"],
                str(int(row["wins"])),
                str(int(row["matches"])),
                f"{float(row['win_rate']):.3f}",
                f"{float(row['price/match']):.6f}",
                f"{float(row['tokens/match']):.2f}",
            ]
            for _, row in df.iterrows()
        ]
        write_table_svg(
            get_ranking_svg_path(suffix),
            title,
            ranking_headers,
            ranking_rows,
        )

    export_ranking(ranking_non_io_df, None, "LLM Win Ranking")
    export_ranking(ranking_io_df, "iointelligence", "LLM Win Ranking (IOIntelligence Only)")

    # --- クロステーブル ---
    cross_grouped_df = (
        cross_df.groupby(["row_model", "col_model"]).sum().reset_index()
        if not cross_df.empty
        else cross_df
    )
    models_non_io = sorted(
        [m for m in models if model_providers.get(m) != "iointelligence"]
    )
    models_io = sorted([m for m in models if model_providers.get(m) == "iointelligence"])

    def export_cross_table(model_list: list[str], suffix: str | None, title: str) -> None:
        cross_headers = ["Model"] + model_list
        filtered_df = (
            cross_grouped_df[
                cross_grouped_df["row_model"].isin(model_list)
                & cross_grouped_df["col_model"].isin(model_list)
            ]
            if (not cross_grouped_df.empty and model_list)
            else cross_grouped_df
        )
        cross_rows: list[list[str]] = []
        diagonal_cells: set[tuple[int, int]] = set()
        for row_idx, row_model in enumerate(model_list):
            row_list: list[str] = [row_model]
            for col_offset, col_model in enumerate(model_list, start=1):
                if row_model == col_model:
                    row_list.append("-")
                    diagonal_cells.add((row_idx, col_offset))
                    continue
                if filtered_df.empty:
                    row_list.append("")
                    continue
                cell = filtered_df[
                    (filtered_df["row_model"] == row_model)
                    & (filtered_df["col_model"] == col_model)
                ]
                if cell.empty:
                    row_list.append("")
                else:
                    wins = int(cell["wins"].iloc[0])
                    losses = int(cell["losses"].iloc[0])
                    draws = int(cell["draws"].iloc[0])
                    total_matches = wins + losses + draws
                    if total_matches == 0:
                        row_list.append("0/0 (0.000)")
                    else:
                        win_rate = wins / total_matches
                        row_list.append(f"{wins}/{total_matches} ({win_rate:.3f})")
            cross_rows.append(row_list)

        write_csv(get_cross_table_csv_path(suffix), cross_headers, cross_rows)
        write_table_svg(
            get_cross_table_svg_path(suffix),
            title,
            cross_headers,
            cross_rows,
            vertical_lines=True,
            diagonal_cells=diagonal_cells,
        )

    export_cross_table(models_non_io, None, "Matchup Cross Table (Wins / Matches)")
    export_cross_table(models_io, "iointelligence", "Matchup Cross Table (IOIntelligence Only)")

    # --- コスト集計 ---
    cost_df = ranking_non_io_df.copy()
    cost_df["price/match"] = cost_df.apply(
        lambda r: (float(r["total_price"]) / r["matches"]) if r["matches"] else 0.0,
        axis=1,
    )
    cost_df["tokens/match"] = cost_df.apply(
        lambda r: (float(r["total_tokens"]) / r["matches"]) if r["matches"] else 0.0,
        axis=1,
    )
    cost_df = cost_df.sort_values("price/match", ascending=False)
    cost_df.to_csv(get_cost_csv_path(), index=False)

    cost_headers = [
        "model",
        "price/match",
        "total_price",
        "token/match",
        "total_tokens",
        "wins/matches",
        "win_rate",
    ]
    cost_rows = []
    for _, row in cost_df.iterrows():
        matches = int(row["matches"])
        wins = int(row["wins"])
        wins_matches = f"{wins}/{matches}" if matches else f"{wins}/0"
        cost_rows.append(
            [
                row["model"],
                f"{float(row['price/match']):.6f}",
                f"{float(row['total_price']):.6f}",
                f"{float(row['tokens/match']):.2f}",
                str(int(row["total_tokens"])),
                wins_matches,
                f"{float(row['win_rate']):.3f}",
            ]
        )

    write_table_svg(
        get_cost_table_svg_path(),
        "Cost Metrics (sorted by cost per match)",
        cost_headers,
        cost_rows,
        vertical_lines=True,
    )

    write_scatter_svg(
        get_cost_performance_svg_path(),
        "Cost Performance",
        cost_df,
        x_col="win_rate",
        y_col="price/match",
        model_col="model",
        x_label="Win rate",
        y_label="Cost per match",
        point_colors=True,
        x_limit=(0, 1.0),
    )

def get_today_oai_cost() ->float:
    today_cost = 0.0
    for result_path in get_result_paths():
        stat = os.stat(result_path)
        file_date = time.localtime(stat.st_mtime)
        today = time.localtime()
        if (file_date.tm_year, file_date.tm_mon, file_date.tm_mday) != (today.tm_year, today.tm_mon, today.tm_mday):
            continue
        with open(result_path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
        for side in ('A','B'):
            side_data = data.get(side, {})
            if side_data.get('provider') == 'openai':
                today_cost += side_data.get('usage', {}).get('total_price', 0.0)
    return today_cost

def main():
    
    overwrite = False
    clist = [
        Config(provider='openai',llm_model='gpt-4o-mini'),
        Config(provider='openai',llm_model='gpt-4o'),
        Config(provider='openai',llm_model='gpt-4.1-mini'),
        Config(provider='openai',llm_model='gpt-4.1'),
        Config(provider='openai',llm_model='gpt-5-mini'),
        Config(provider='openai',llm_model='gpt-5'),
        # Config(provider='openai',llm_model='gpt-5-M'),
        Config(provider='anthropic',llm_model='Claude-Haiku-3'),
        Config(provider='anthropic',llm_model='Claude-Haiku-3.5'),
        Config(provider='anthropic',llm_model='Claude-Sonnet-4'),
        Config(provider='anthropic',llm_model='Claude-Sonnet-4.5'),
        #Config(provider='anthropic',llm_model='Claude-Sonnet-4-M'),
        Config(provider='gemini',llm_model='gemini-2.5-flash-lite'),
        Config(provider='gemini',llm_model='gemini-2.5-flash'),
    ]
    clist = [
        Config(provider='iointelligence',llm_model='gpt-oss-20b'),
        Config(provider='iointelligence',llm_model='gpt-oss-120b'),
        #
        Config(provider='iointelligence',llm_model='deepseek-r1'),
        #
        Config(provider='iointelligence',llm_model='qwen3-coder-480b-a35b-instruct'),
        Config(provider='iointelligence',llm_model='qwen3-next-80b-a3b-instruct'),
        Config(provider='iointelligence',llm_model='qwen3-235b-a22b-thinking-2507'),
        Config(provider='iointelligence',llm_model='qwen2-5-vl-32b'),
        #
        Config(provider='iointelligence',llm_model='llama4-17b'),
        Config(provider='iointelligence',llm_model='llama3-3-70b'),
        Config(provider='iointelligence',llm_model='llama3-2-90b-vision-instruct'),
    ]
    vslist = []
    for c1 in clist:
        for c2 in clist:
            if c1.llm_model == c2.llm_model:
                continue
            vslist.append( (c1,c2) )

    ccclist = []
    while len(vslist)>0:
        i = 0
        mcount = {}
        while( i<len(vslist)):
            c1,c2 = vslist[i]
            if c1.llm_model not in mcount and c2.llm_model not in mcount:
                mcount[c1.llm_model] = 1
                mcount[c2.llm_model] = 1
                ccclist.append( (c1,c2) )
                vslist.pop(i)
            else:
                i+=1

    for c1,c2 in ccclist:
        print(f"{c1.llm_model} vs {c2.llm_model}")
    print("-----")

    n=1
    newruns = 0
    today_oai_cost = get_today_oai_cost()
    print(f"Today cost so far: ${today_oai_cost:.4f}")
    if n>0:
        for c1,c2 in ccclist:
                if c1.llm_model == c2.llm_model:
                    continue
                if today_oai_cost>4.0:
                    print("Today cost exceeded $4.0, stop further runs.")
                    break
                print(f"\n\n=== Competition: {c1.llm_model} vs {c2.llm_model} ===")
                competition = LLMCompetition(c1,c2)
                if competition.eval( overwrite=overwrite):
                    print("Results:")
                    results()
                    newruns += 1
                    today_oai_cost = get_today_oai_cost()
                    print(f"Today cost so far: ${today_oai_cost:.4f}")
    if newruns==0:
        print("Results:")
        results()

if __name__ == "__main__":
    main()
