
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

import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')  # ヘッドレス環境でもSVG出力可能にする
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None
    Line2D = None

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover
    adjust_text = None

SVG_FONT_FAMILY = "monospace"
SVG_BG_COLOR = "#ffffff"
SVG_HEADER_BG = "#e5e7eb"
SVG_ROW_BG = "#f9fafb"
SVG_BORDER_COLOR = "#4b5563"
SVG_DIAGONAL_COLOR = "#9ca3af"


CLAUDE_COLOR = "#ff8c00"  # オレンジ
GEMINI_COLOR = "#0015FF"  # 濃い青
GPT_COLOR = "#000000"  # 黒
OTHER_COLOR = "#555555"  # 区分外は濃いグレー


COLOR_MAP = {
    "Claude": CLAUDE_COLOR,
    "Gemini": GEMINI_COLOR,
    "GPT": GPT_COLOR,
    "Other": OTHER_COLOR,
}

MODEL_PREFIXES_TO_HIDE = ("claude-", "gemini-")

def resolve_model_category(model: str | None) -> str:
    model_lower = (model or "").lower()
    if "claude" in model_lower:
        return "Claude"
    if model_lower.startswith("gemini"):
        return "Gemini"
    if model_lower.startswith("gpt"):
        return "GPT"
    return "Other"

def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def strip_model_prefix(label: str | None) -> str:
    """散布図などでの表示用に既知のモデル接頭辞を削除する。"""
    text = label or ""
    lower = text.lower()
    for prefix in MODEL_PREFIXES_TO_HIDE:
        if lower.startswith(prefix):
            return text[len(prefix):]
    return text

def write_table_svg(
    path: str,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    vertical_lines: bool = False,
    diagonal_cells: set[tuple[int, int]] | None = None,
) -> None:
    if not headers:
        return
    ensure_parent_dir(path)
    CHAR_WIDTH = 7
    CELL_PADDING_X = 12
    COL_MIN_WIDTH = 80
    BASE_ROW_HEIGHT = 28
    LINE_HEIGHT = 16
    HEADER_EXTRA_PADDING = 8
    DATA_EXTRA_PADDING = 8
    margin_x = 24
    title_height = 36

    def wrap_label(value: str) -> str:
        text = value or ""
        ltext = text.lower()
        prefixes = ("claude-", "gemini-")
        for prefix in prefixes:
            if ltext.startswith(prefix):
                text = text[:len(prefix)] + '\n' + text[len(prefix):]
        return text

    def to_lines(value: str) -> list[str]:
        wrapped = wrap_label(str(value) if value is not None else "")
        lines = [line for line in wrapped.split("\n")]
        if not lines:
            return [""]
        return lines

    num_cols = len(headers)
    header_lines = [to_lines(h) for h in headers]
    formatted_rows: list[list[list[str]]] = []
    for row in rows:
        formatted_row: list[list[str]] = []
        for col_idx in range(num_cols):
            if col_idx < len(row):
                formatted_row.append(to_lines(row[col_idx]))
            else:
                formatted_row.append([""])
        formatted_rows.append(formatted_row)

    col_widths: list[int] = []
    for col_idx in range(num_cols):
        longest = 0
        candidate_groups = [header_lines[col_idx]]
        for row in formatted_rows:
            candidate_groups.append(row[col_idx])
        for group in candidate_groups:
            for line in group:
                longest = max(longest, len(line))
        col_width = max(COL_MIN_WIDTH, longest * CHAR_WIDTH + CELL_PADDING_X * 2)
        col_widths.append(col_width)

    header_line_count = max((len(lines) for lines in header_lines), default=1)
    header_height = max(BASE_ROW_HEIGHT, header_line_count * LINE_HEIGHT + HEADER_EXTRA_PADDING)

    row_heights: list[int] = []
    for row in formatted_rows:
        max_lines = max((len(cell_lines) for cell_lines in row), default=1)
        row_heights.append(max(BASE_ROW_HEIGHT, max_lines * LINE_HEIGHT + DATA_EXTRA_PADDING))

    width = margin_x * 2 + sum(col_widths)
    height = title_height + header_height + sum(row_heights) + margin_x

    def cell_position(column: int) -> int:
        return margin_x + sum(col_widths[:column])

    column_edges: list[int] = [margin_x]
    for width_value in col_widths:
        column_edges.append(column_edges[-1] + width_value)

    diagonal_lookup = diagonal_cells or set()

    def render_text(fp, lines: list[str], x_left: int, y_top: float, box_height: int, *, bold: bool = False) -> None:
        if not lines:
            lines = [""]
        total_text_height = LINE_HEIGHT * len(lines)
        y_offset = y_top + max(0, (box_height - total_text_height) / 2.0) + 12
        weight_attr = " font-weight='bold'" if bold else ""
        fp.write(
            f"  <text x='{x_left}' y='{y_offset}' text-anchor='start'{weight_attr}>")
        fp.write("\n")
        for idx, line in enumerate(lines):
            escaped = escape(line)
            if idx == 0:
                fp.write(f"    {escaped}\n")
            else:
                dy = LINE_HEIGHT
                fp.write(f"    <tspan x='{x_left}' dy='{dy}'>{escaped}</tspan>\n")
        fp.write("  </text>\n")

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
            f"font-family='{SVG_FONT_FAMILY}' font-size='14'>\n"
        )
        fp.write(f"  <rect x='0' y='0' width='{width}' height='{height}' fill='{SVG_BG_COLOR}'/>\n")
        fp.write(
            f"  <text x='{width/2}' y='{title_height/2 + 6}' text-anchor='middle' font-size='18' font-weight='bold'>"
            f"{escape(title)}</text>\n"
        )

        # Header row background
        fp.write(
            f"  <rect x='{margin_x}' y='{title_height}' width='{sum(col_widths)}' height='{header_height}' fill='{SVG_HEADER_BG}' stroke='{SVG_BORDER_COLOR}'/>\n"
        )
        for idx, lines in enumerate(header_lines):
            x = cell_position(idx) + CELL_PADDING_X
            render_text(fp, lines, x, title_height, header_height, bold=True)

        # Table grid and rows
        current_y = title_height + header_height
        for row_index, (row_lines, row_height) in enumerate(zip(formatted_rows, row_heights)):
            fill = SVG_ROW_BG if (row_index + 1) % 2 == 1 else SVG_BG_COLOR
            fp.write(
                f"  <rect x='{margin_x}' y='{current_y}' width='{sum(col_widths)}' height='{row_height}' fill='{fill}' stroke='{SVG_BORDER_COLOR}'/>\n"
            )
            for col_idx, cell_lines in enumerate(row_lines):
                left = column_edges[col_idx]
                right = column_edges[col_idx + 1]
                top = current_y
                bottom = current_y + row_height
                if (row_index, col_idx) in diagonal_lookup:
                    fp.write(
                        f"  <line x1='{left}' y1='{top}' x2='{right}' y2='{bottom}' stroke='{SVG_DIAGONAL_COLOR}' stroke-width='2'/>\n"
                    )
                x = left + CELL_PADDING_X
                render_text(fp, cell_lines, x, current_y, row_height)
            current_y += row_height

        if vertical_lines:
            table_top = title_height
            table_bottom = title_height + header_height + sum(row_heights)
            for edge in column_edges[1:-1]:
                fp.write(
                    f"  <line x1='{edge}' y1='{table_top}' x2='{edge}' y2='{table_bottom}' stroke='{SVG_BORDER_COLOR}' stroke-width='1'/>\n"
                )

        fp.write("</svg>\n")

def write_scatter_svg(
    output_path: str,
    title: str,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    model_col: str = "Model",
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    x_limit: tuple[float, float] | None = (0, 100),
    point_colors: bool = False,
    x_reference: float | None = None,
    y_limit: tuple[float, float] | None = None,
) -> None:
    valid_rows = df[[x_col, y_col, model_col]].dropna(subset=[x_col, y_col])
    points: list[tuple[str, float, float]] = [
        (row[model_col], float(row[x_col]), float(row[y_col])) for _, row in valid_rows.iterrows()
    ]

    pt_write_scatter_svg(
        output_path,
        title,
        points,
        x_label if x_label else x_col,
        y_label if y_label else y_col,
        point_colors=point_colors,
        x_lim=x_limit,
        x_reference=x_reference,
        y_lim=y_limit,
    )

def pt_write_scatter_svg(
    path: str,
    title: str,
    points: list[tuple[str, float, float]],
    x_label: str,
    y_label: str,
    *,
    point_colors: bool = False,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    x_reference: float | None = None,
) -> None:
    if plt is None or adjust_text is None:
        raise RuntimeError("matplotlib または adjustText が見つかりません。`pip install matplotlib adjustText` を実行してください。")

    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_lim is not None:
        ax.set_xlim(*x_lim)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if x_reference is not None:
        # 精度100%位置の視認性を高めるための補助線
        ax.axvline(x_reference, color='#1f2937', linestyle='-', linewidth=1.0, alpha=0.6)

    if not points:
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
    else:
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        if point_colors:
            cc = [COLOR_MAP.get(resolve_model_category(p[0]), COLOR_MAP["Other"]) for p in points]
            ax.scatter(xs, ys, c=cc, s=60, alpha=0.85, edgecolors='white', linewidths=0.5)
        else:
            ax.scatter(xs, ys, c=ys, cmap='viridis', s=60, alpha=0.85, edgecolors='white', linewidths=0.5)
        texts = [
            ax.text(x, y, strip_model_prefix(label), fontsize=10, ha='center', va='center')
            for label, x, y in points
        ]
        adjust_text(
            texts,
            ax=ax,
            expand_points=(1.2, 1.4),
            expand_text=(1.05, 1.2),
            arrowprops=dict(arrowstyle='-', color='#4b5563', lw=0.6),
            force_text=0.5,
            force_points=0.2,
        )
        ax.grid(True, linestyle='--', alpha=0.3)

    if point_colors and points:
        if Line2D is None:
            raise RuntimeError("matplotlib の Line2D が利用できないため凡例を描画できません。")
        legend_items = [
            (category, COLOR_MAP[category])
            for category in set([resolve_model_category(p[0]) for p in points])
            if category in COLOR_MAP
        ]
        handles = [
            Line2D(
                [0],
                [0],
                marker='o',
                linestyle='',
                markerfacecolor=color,
                markeredgecolor='#ffffff',
                markeredgewidth=0.5,
                label=label,
            )
            for label, color in legend_items
        ]
        legend_title = "Model Category"
        ax.legend(handles=handles, title=legend_title)

    fig.tight_layout()
    fig.savefig(path, format='svg')
    plt.close(fig)
