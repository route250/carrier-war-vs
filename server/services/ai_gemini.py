"""
GeminiベースのLLMボット実装（PvPエンジン上で動作）。

このモジュールは Gemini 依存部のみを提供し、共通ロジックは
`server/services/ai_llm_base.py` の `LLMBase` に集約した。
"""

from __future__ import annotations

import json
import os
import time
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, List
from pydantic import BaseModel

from server.services.ai_llm_base import (
    LLMError, LLMRateLimitError,
    LLMBase,
    LLMBaseConfig,
    ROLE_SYSTEM, ROLE_USER, ROLE_AI,
    LLMTokenUsage,
    exists_env,
    ResponseModel, Coordinate
)
from server.schemas import PlayerOrders, AIModel, AIProvider, AIListResponse

# gemini SDK
try:
    from google import genai
    from google.genai.types import Content, Part
    from google.genai.types import GenerateContentResponse, CountTokensResponse
    from google.genai.errors import APIError
    USE_GEMINI = True
except Exception:  # ランタイム環境により未導入の可能性に備える
    USE_GEMINI = False  # type: ignore

DEFAULT_MODEL = "gemini-2.5-flash"

MODELS = AIProvider(name="Gemini", models=[

    AIModel(name="gemini-2.0-flash-lite", model="gemini-2.0-flash-lite",
            max_input_tokens=1048576, input_strategy='summarize', max_output_tokens=8192,
            input_price=0.075, cached_price=0.0, output_price=0.30,
            tpm=250000, rpm=10),

    AIModel(name="gemini-2.0-flash", model="gemini-2.0-flash",
            max_input_tokens=1048576, input_strategy='summarize', max_output_tokens=8192,
            input_price=0.10, cached_price=0.025, output_price=0.40,
            tpm=250000, rpm=10),
    # AIModel(name="gemini-2.0-pro", model="gemini-2.0-pro"),

    AIModel(name="gemini-2.5-flash-lite", model="gemini-2.5-flash-lite",
            max_input_tokens=1048576, input_strategy='summarize', max_output_tokens=65536,
            input_price=0.10, cached_price=0.025, output_price=0.40,
            tpm=250000, rpm=10),

    AIModel(name="gemini-2.5-flash", model="gemini-2.5-flash",
            max_input_tokens=1048576, input_strategy='summarize', max_output_tokens=65536,
            input_price=0.30, cached_price=0.075, output_price=2.50,
            tpm=250000, rpm=10),

    AIModel(name="gemini-2.5-pro", model="gemini-2.5-pro",
            max_input_tokens=1048576, input_strategy='summarize', max_output_tokens=65536,
            input_price=1.25, cached_price=0.31, output_price=10.00,
            tpm=250000, rpm=10),

])

@dataclass
class GeminiConfig(LLMBaseConfig):
    model: str = MODELS.default().model


def convert_exception( aex: Exception) -> LLMError:
    if isinstance(aex, APIError):
        ex: APIError = aex # type: ignore
        details = ex.details.get('error',{}).get('details',[])
        if ex.code == 429:
            wait = None
            for detail_info in details if isinstance(details,list) else []:
                if isinstance(detail_info,dict) and 'retryDelay' in detail_info:
                    delay_str = detail_info.get('retryDelay')
                    if isinstance(delay_str,str):
                        if delay_str.endswith('s'):
                            delay_str = delay_str[:-1]
                        try:
                            wait = float(delay_str)
                        except Exception:
                            pass
            return LLMRateLimitError( ex.message or "rate limit error", retry_after=wait)
        elif ex.code == 503 or ex.code == 500:
            return LLMRateLimitError( ex.message or "service unavailable", retry_after=None)
        if ex.message == 'The model is overloaded. Please try again later.':
            return LLMRateLimitError( ex.message or "rate limit error", retry_after=None)

    return LLMError(ex)

class CarrierBotGemini(LLMBase):
    """Gemini LLM を用いるボット（LLMBase 継承）。"""

    def __init__(
        self,
        store,
        match_id: str,
        *,
        name: str|None = None,
        config: Optional[GeminiConfig] = None,
    ) -> None:
        ai_model = MODELS.get_model(name=name,config=config)
        assert ai_model is not None, f"Unknown Gemini model: {name}"
        config = config or GeminiConfig()
        config.model = ai_model.model
        super().__init__(store=store, match_id=match_id, ai_model=ai_model, config=config)
        # サブクラス固有
        self.use_output_format:bool = True
        self._config: GeminiConfig = config
        self._client: genai.Client|None = None

    @staticmethod
    def get_model_names() -> AIProvider:
        return MODELS

    # --- Gemini依存部 ---
    def _ensure_client_ready(self) -> bool:
        if self._client is not None:
            return True
        if USE_GEMINI is None:
            return False
        if not self._config.api_key and not exists_env("GEMINI_API_KEY"):
            return False
        try:
            self._client = genai.Client(api_key=self._config.api_key)
            return True
        except Exception:
            self._client = None
            return False

    def convert_to(self, orig:list[dict[str, str]]) -> tuple[Content|None, list[Content]]:
        system_prompt = []
        xmsgs:list[Content] = []
        for m in orig:
            if m['role'] == ROLE_SYSTEM:
                system_prompt.append(m['content'])
            elif m['role'] == ROLE_AI:
                xmsgs.append(Content( role='model', parts=[Part(text=m.get('content',''))] ))
            else:
                xmsgs.append(Content( role='user', parts=[Part(text=m.get('content',''))] ))
        system_instruction = None
        if len(system_prompt)>0:
            system_instruction = Content( role='system', parts=[Part(text=a) for a in system_prompt] )
        return system_instruction, xmsgs

    def count_tokens(self, msgs: list[dict[str, str]], *, output_format: type[BaseModel]|None=None) -> int:
        """メッセージのトークン数を数える。"""
        if not msgs or self._client is None:
            return 0

        system_prompt, xmsgs = self.convert_to(msgs)
        if system_prompt:
            xmsgs.insert(0, system_prompt)
        config:genai.types.CountTokensConfig|None = None
        if output_format:
            xmsgs.append( Content(role='user', parts=[Part(text=json.dumps(output_format.model_json_schema()))] ) )

        try:
            response: CountTokensResponse = self._client.models.count_tokens(
                model=self.aimodel.model,
                contents=xmsgs,  # type: ignore
                config=config)
            total_tokens = response.total_tokens if response and response.total_tokens else 0
            return total_tokens
        except Exception as ex:
            print(f"Gemini count_tokens error: {ex}")
            pass

        return 0

    def LLM(self, msgs: list[dict[str,str]], *, output_format: type[BaseModel]|None=None) -> str:
        if self._client is None:
            raise RuntimeError("Gemini client not initialized")

        system_prompt, xmsgs = self.convert_to(msgs)

        config:genai.types.GenerateContentConfig= genai.types.GenerateContentConfig(
            system_instruction=system_prompt
        )
        if output_format:
            config.response_mime_type= "application/json"
            config.response_schema= output_format

        predict_tokens = self.count_tokens(msgs, output_format=output_format)

        response: GenerateContentResponse|None = None
        usage: LLMTokenUsage|None = None
        ntry = 0
        while True:
            t = self._rating_info.tokens+predict_tokens
            if self.aimodel.tpm and t >= self.aimodel.tpm:
                print(f"Gemini token limit: wait... {t} / {self.aimodel.tpm}")
                time.sleep(10.0)
                continue
            elif self.aimodel.rpm and self._rating_info.count >= self.aimodel.rpm:
                print(f"Gemini rate limit: wait... {self._rating_info.count} / {self.aimodel.rpm}")
                time.sleep(10.0)
                continue

            try:
                ntry += 1
                response = self._client.models.generate_content(
                    model=self.aimodel.model,
                    contents=xmsgs, # type: ignore
                    config=config
                )
                break
            except Exception as ex:
                llmerror = convert_exception(ex)
                if isinstance(llmerror, LLMRateLimitError):
                    if llmerror.retry_after and llmerror.retry_after>0:
                        wait = float(int(llmerror.retry_after+1))
                    else:
                        if ntry >= 6:
                            raise llmerror
                        wait = 90.0
                    print(f"Gemini rate limit: {llmerror}. wait {wait} sec and retry...")
                    time.sleep(wait)
                    continue
                raise llmerror

        if response is not None and response.parsed and isinstance(response.parsed, output_format.__class__):
                assistant_text = json.dumps( response.parsed.model_dump(), ensure_ascii=False )
        else:
            assistant_text = response.text

        try:
            um = response.usage_metadata
            input_tokens = um.prompt_token_count if um else 0
            output_tokens = um.candidates_token_count if um else 0
            total_tokens = um.total_token_count if um else 0
            cache_read_tokens = um.cached_content_token_count if um else 0
            usage = LLMTokenUsage(
                prompt=input_tokens or 0,
                completion=output_tokens or 0,
                reasoning=0,
                cache_read=cache_read_tokens or 0,
            )
            self._rating_info.add(time=time.time(), tokens=total_tokens or 0)
        except Exception as ex:
            pass

        self._set_last_usage_tokens(usage)

        if not assistant_text:
            raise RuntimeError("empty response from model")

        return assistant_text

    def _diag_model_name(self) -> str:  # type: ignore[override]
        return self._config.model
