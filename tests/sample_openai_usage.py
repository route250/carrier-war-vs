"""OpenAI課金使用量の当日分を取得するサンプルスクリプト。"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError


API_URL = "https://api.openai.com/v1/organization/costs"


def to_dt( utc_ts:int ) -> datetime:
    """UTC秒からUTC日時へ変換する。"""
    return datetime.fromtimestamp(utc_ts, tz=timezone.utc)


@dataclass
class UsageWindow:
    """問い合わせ対象となる開始・終了時刻（UTC秒）。"""

    start_ts: int
    end_ts: int
    label: str


def compute_today_window( days:int ) -> UsageWindow:
    """ローカルタイムゾーンの「今日」をUTC秒レンジへ変換する。"""
    end_local = datetime.now().astimezone()
    aaa = end_local - timedelta(days=days if days > 0 else 0)
    start_local = aaa.replace(hour=3, minute=0, second=0, microsecond=0)
    start_ts = int(start_local.astimezone(timezone.utc).timestamp())
    end_ts = int(end_local.astimezone(timezone.utc).timestamp())
    label = start_local.strftime("%Y-%m-%d")
    return UsageWindow(start_ts=start_ts, end_ts=end_ts, label=label)


def to_decimal(value: object) -> float:
    """数値変換ができない場合でも0を返して継続する。"""
    if isinstance(value, int|float):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return 0

"""

  "object": "page",
  "has_more": false,
  "next_page": null,
  "data": [
    {
      "object": "bucket",
      "start_time": 1759017600,
      "end_time": 1759104000,
      "results": [
        {
          "object": "organization.costs.result",
          "amount": {
            "value": 10.1425634,
            "currency": "usd"
          },
          "line_item": null,
          "project_id": null,
          "organization_id": "org-YjkYStpDHMfCmoB39rPJfHay"
        }
      ]
    }
  ]
}

"""
class CostAmount(BaseModel):
    """コスト金額を表すモデル。"""

    model_config = ConfigDict(extra="ignore")

    value: float
    currency: str


class CostResult(BaseModel):
    """バケット内のコスト結果。"""

    model_config = ConfigDict(extra="ignore")

    object: str
    amount: CostAmount
    line_item: str|None = None
    project_id: str|None = None
    organization_id: str|None = None


class CostBucket(BaseModel):
    """時間帯ごとのコスト情報。"""

    model_config = ConfigDict(extra="ignore")

    object: str
    start_time: int
    end_time: int
    results: list[CostResult] = Field(default_factory=list)


class CostPage(BaseModel):
    """`/v1/organization/costs` のレスポンス全体。"""

    model_config = ConfigDict(extra="ignore")

    object: str
    has_more: bool
    next_page: str|None = None
    data: list[CostBucket] = Field(default_factory=list)


def parse_costs_payload(payload: dict[str, object]) -> CostPage:
    """レスポンス辞書を `CostPage` へバリデーション付きで変換する。"""

    return CostPage.model_validate(payload)


def summarize_total_cost_from_model(page: CostPage) -> float:
    """Pydanticモデルから合計コスト(USD)を算出する。"""

    total:float = 0.0
    for bucket in page.data:
        for result in bucket.results:
            total += result.amount.value
    return total


def build_common_headers(*, include_usage_beta: bool = False) -> dict[str, str]:
    api_key = os.getenv("OPENAI_ADMIN_KEY")
    if not api_key:
        raise RuntimeError("環境変数 OPENAI_ADMIN_KEY が未設定です")

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
    }

    if include_usage_beta:
        headers["OpenAI-Beta"] = "usage=v1"

    org = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")
    if org:
        headers["OpenAI-Organization"] = org

    return headers


def fetch_usage(window: UsageWindow, page=None) -> dict[str, object]:
    headers = build_common_headers(include_usage_beta=True)

    params = {
        "start_time": window.start_ts,
        "end_time": window.end_ts,
    }
    if page:
        params["page"] = page

    with httpx.Client(timeout=30.0) as client:
        response = client.get(API_URL, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

def main() -> int:
    window = compute_today_window(3)
    results: list[CostBucket] = []
    next_page = None
    try:
        while True:
            payload = fetch_usage(window, page=next_page)
            cost_page = parse_costs_payload(payload)
            # 有効なデータだけ集める
            if isinstance(cost_page.data,list):
                for bucket in cost_page.data:
                    if isinstance(bucket.results,list) and len(bucket.results) > 0:
                        results.append(bucket)
            next_page = payload.get("next_page")
            if not next_page:
                break
    except httpx.HTTPStatusError as exc:
        print(f"[error] APIリクエストが失敗しました: {exc.response.status_code} {exc.response.text}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    total: float = 0.0
    for bucket in results:
        st = to_dt(bucket.start_time).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        et = to_dt(bucket.end_time).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"バケット: {st} – {et}")
        for result in bucket.results:
            print(f"    {result.amount.value}")
            total += result.amount.value
    print(f"コスト合計(USD): ${total}")



    return 0


if __name__ == "__main__":
    sys.exit(main())
