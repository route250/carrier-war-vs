"""
io.netベースのLLMボット実装（PvPエンジン上で動作）。

このモジュールは io.net 依存部のみを提供し、共通ロジックは
`server/services/ai_llm_base.py` の `LLMBase` に集約した。
"""

from __future__ import annotations

import os

# If tests or imports run this module directly, ensure project-local `config.env` is loaded so
# environment variables like OPENAI_API_KEY are available during import-time.
try:
    from pathlib import Path
    from dotenv import load_dotenv
    # ai_openai.py is at server/services; project root is two levels up
    p = Path(__file__).resolve().parents[2] / "config.env"
    if p.exists():
        load_dotenv(dotenv_path=str(p))
except Exception:
    # dotenv may be unavailable in some minimal test environments; ignore if not present
    pass

from server.services.ai_llm_base import LLMBaseConfig
from server.schemas import AIModel, AIProvider
from server.services.ai_openai import CarrierBotOpenAI

IOINTELLIGENCE_BASE_URL = "https://api.intelligence.io.solutions/api/v1/"

IOINTELLIGENCE_MODELS = AIProvider(name="iointelligence", models=[

            # gpt-oss シリーズ
            AIModel(name="gpt-oss-120b", model="openai/gpt-oss-120b",
                    max_input_tokens=128000, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),
            AIModel(name="gpt-oss-20b", model="openai/gpt-oss-20b",
                    max_input_tokens=64000, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),

            # deepseek シリーズ
            AIModel(name="deepseek-r1", model="deepseek-ai/DeepSeek-R1-0528",
                    max_input_tokens=128000, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),

            # Qwen シリーズ
            AIModel(name="qwen3-coder-480b-a35b-instruct", model="Intel/Qwen3-Coder-480B-A35B-Instruct-int4-mixed-ar",
                    max_input_tokens=106000, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),
            AIModel(name="qwen3-next-80b-a3b-instruct", model="Qwen/Qwen3-Next-80B-A3B-Instruct",
                    max_input_tokens=262144, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),
            AIModel(name="qwen3-235b-a22b-thinking-2507", model="Qwen/Qwen3-235B-A22B-Thinking-2507",
                    max_input_tokens=262144, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),
            AIModel(name="qwen2-5-vl-32b", model="Qwen/Qwen2.5-VL-32B-Instruct",
                    max_input_tokens=16000, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),

            # llama シリーズ
            AIModel(name="llama4-17b", model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    max_input_tokens=430000, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),
            AIModel(name="llama3-3-70b", model="meta-llama/Llama-3.3-70B-Instruct",
                    max_input_tokens=128000, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),
            AIModel(name="llama3-2-90b-vision-instruct", model="meta-llama/Llama-3.2-90B-Vision-Instruct",
                    max_input_tokens=128000, max_output_tokens=8000,
                    input_price=0.05, cached_price=0.005, output_price=0.4,
                    base_url=IOINTELLIGENCE_BASE_URL,
                    input_strategy="truncate", output_format="json_text"),

            # mistralai/Mistral-Nemo-Instruct-2407
            # mistralai/Magistral-Small-2506
            # mistralai/Devstral-Small-2505
            # mistralai/Mistral-Large-Instruct-2411

            # swiss-ai/Apertus-70B-Instruct-2509
            # LLM360/K2-Think
            # BAAI/bge-multilingual-gemma2

        ])

class CarrierBotIOIntelligence(CarrierBotOpenAI):
    """OpenAI LLM を用いるボット（LLMBase 継承）。"""

    def __init__(
        self,
        store,
        match_id: str,
        *,
        name: str|None = None,
        ai_model: AIModel|None = None,
        config: LLMBaseConfig|None = None,
    ) -> None:
        ai_model = IOINTELLIGENCE_MODELS.get_model(name=name,config=config)
        assert ai_model is not None, f"Unknown io.net model: {name}"
        super().__init__(store=store, match_id=match_id, name=name, ai_model=ai_model, config=config)
        if not self._config.api_key and os.getenv("IOINTELLIGENCE_API_KEY"):
            self._config.api_key = os.getenv("IOINTELLIGENCE_API_KEY")

    @staticmethod
    def get_model_names() -> AIProvider:
        return IOINTELLIGENCE_MODELS

    @staticmethod
    def get_model(name: str|None = None) -> AIModel|None:
        if name:
            return IOINTELLIGENCE_MODELS.find(name) or None
        return None

    @staticmethod
    def get_default_model() -> AIModel:
        return IOINTELLIGENCE_MODELS.models[0]

