import json
import sys,os
from pathlib import Path
from typing import Iterable

import pandas as pd

if __name__ == '__main__':
    # server/services配下のai_gemini.pyを読み込むため、プロジェクトルート（二階層上）をインポートパスへ追加
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.plotlib import write_scatter_svg, resolve_model_category, COLOR_MAP

def load_summary_table(json_source: str | Path) -> pd.DataFrame:
    """指定ディレクトリまたはファイルからベンチ結果のサマリを集計する。"""

    def iter_files(base: Path) -> Iterable[Path]:
        if base.is_file():
            yield base
            return
        for path in sorted(base.glob("*.json")):
            if path.is_file():
                yield path

    base_path = Path(json_source)
    rows = []
    for json_path in iter_files(base_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {})
        cost = data.get("cost", {})
        rows.append(
            {
                "Provider": data.get("provider"),
                "Model": data.get("model"),
                "Reasoning": data.get("reasoning"),
                "Elapsed": data.get("elapsed"),
                "CellType_correct": summary.get("CellType", {}).get("correct"),
                "CellType_total": summary.get("CellType", {}).get("total"),
                "Count_correct": summary.get("Count", {}).get("correct"),
                "Count_total": summary.get("Count", {}).get("total"),
                "Neighbors_correct": summary.get("Neighbors", {}).get("correct"),
                "Neighbors_total": summary.get("Neighbors", {}).get("total"),
                "Total_correct": summary.get("Total", {}).get("correct"),
                "Total_total": summary.get("Total", {}).get("total"),
                "TotalPrice": cost.get("total_price"),
            }
        )
    return pd.DataFrame(rows)


def draw_scatter(
    output_path: Path,
    title: str,
    df: pd.DataFrame,
    *,
    x_col: str = "AccuracyPercent",
    y_col: str = "TotalPrice",
    x_label: str = "Accuracy (%)",
    y_label: str = "TotalPrice",
    x_limit: tuple[float, float] | None = (0, 100),
) -> None:
    write_scatter_svg(
        str(output_path),
        title,
        df,
        x_col,
        y_col,
        "Model",
        x_label=x_label,
        y_label=y_label,
        point_colors=True,
        x_limit=x_limit,
    )

def plot_bench_results():
    dir = os.path.join('tests', 'bench_map_detect_results')
    df = load_summary_table(dir)
    if df.empty:
        raise FileNotFoundError(f"{dir} 配下に集計可能なJSONが見つかりません")

    output_dir = Path(dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy = (df["Total_correct"] / df["Total_total"]).fillna(0) * 100
    df = df.assign(AccuracyPercent=accuracy)

    print(df.head(30))
    df.to_csv(output_dir / "bench_plot.csv", index=False)

    # 1枚目: 横軸: 正解率(%), 縦軸: TotalPrice、全モデルで描画
    draw_scatter(output_dir / "bench_plot_price.svg", "Accuracy vs Price (All Models)",df)

    # 2枚目: 最高価格の{max_rate}%以下のモデルのみで描画
    max_rate = 10  # 最高価格の10%以下
    max_price = df["TotalPrice"].max()
    if pd.isna(max_price):
        filtered_df = df.iloc[0:0]
    else:
        threshold = max_price * (max_rate / 100)
        filtered_df = df[df["TotalPrice"] <= threshold]

    if not filtered_df.empty:
        draw_scatter(
            output_dir / "bench_plot_price2.svg",
            f"Accuracy vs Price(Models within {max_rate}% of Max Price)",
            filtered_df
        )
    else:
        print(f"最高価格の{max_rate}%以下に該当するモデルが存在しないため bench_plot_price2.svg は生成されませんでした。")

    # 2枚目: 横軸: 正解率(%), 縦軸: Elapsed、点にモデル名を付与
    draw_scatter(
        output_dir / "bench_plot_time.svg",
        "Accuracy vs Elapsed Time (All Models)",
        df,
        y_col="Elapsed",
        y_label="Elapsed Time",
    )

if __name__ == "__main__":
    plot_bench_results()
