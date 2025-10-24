"""
AnthropicベースのLLMボット実装（PvPエンジン上で動作）。

このモジュールは Anthropic 依存部のみを提供し、共通ロジックは
`server/services/ai_llm_base.py` の `LLMBase` に集約した。
"""

from __future__ import annotations

import json
import os
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import httpx
from pydantic import BaseModel


from server.services.ai_llm_base import (
    LLMError, LLMRateLimitError,
    LLMBase,
    LLMBaseConfig,
    LLMTokenUsage,
    ROLE_SYSTEM, ROLE_USER, ROLE_AI,
    exists_env
)
from server.schemas import PlayerOrders, AIModel, AIProvider, AIListResponse


# anthropic SDK
try:
    from anthropic import Anthropic
    from anthropic.types.message_param import MessageParam
    from anthropic.types.message import Message
    from anthropic.types.content_block import ContentBlock
    from anthropic.types.text_block import TextBlock
    from anthropic.types.text_block_param import TextBlockParam
    from anthropic.types.usage import Usage
    from anthropic.types.thinking_config_enabled_param import ThinkingConfigEnabledParam
    from anthropic.types.thinking_config_disabled_param import ThinkingConfigDisabledParam
    from anthropic._types import NotGiven
    from anthropic import RateLimitError
    USE_ANTHROPIC = True
except Exception:  # ランタイム環境により未導入の可能性に備える
    USE_ANTHROPIC = False  # type: ignore

DEFAULT_MODEL = "claude-3-haiku-20240307"

# claude-opus-4-1-20250805
# claude-opus-4-20250514
# claude-sonnet-4-20250514
# claude-3-7-sonnet-20250219
# claude-3-5-haiku-20241022
# claude-3-haiku-20240307

MODELS = AIProvider(name="Anthropic", models=[
            AIModel(name="Claude-Haiku-3", model="claude-3-haiku-20240307",
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=4096,
                    input_price=0.25, cached_price=0.03, cache_write_price=0.3, output_price=1.25,
                    rpm=1000,tpm=100000),
            AIModel(name="Claude-Haiku-3.5", model="claude-3-5-haiku-20241022",
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=8192,
                    input_price=0.8, cached_price=0.08, cache_write_price=1.0, output_price=4.0,
                    rpm=1000,tpm=100000),
            AIModel(name="Claude-Haiku-4.5", model="claude-haiku-4-5-20251001",
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=64000,
                    input_price=1, cached_price=0.1, cache_write_price=1.25, output_price=5.0,
                    rpm=1000,tpm=100000),
            AIModel(name="Claude-Sonnet-3.7", model="claude-3-7-sonnet-20250219",
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=64000,
                    input_price=3.0, cached_price=0.3, cache_write_price=3.75, output_price=15.0,
                    rpm=1000,tpm=40000),
            AIModel(name="Claude-Sonnet-4", model="claude-sonnet-4-20250514",
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=64000,
                    input_price=3.0, cached_price=0.3, cache_write_price=3.75, output_price=15.0,
                    rpm=1000,tpm=450000),
            AIModel(name="Claude-Sonnet-4-M", model="claude-sonnet-4-20250514",reasoning='medium',
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=64000,
                    input_price=3.0, cached_price=0.3, cache_write_price=3.75, output_price=15.0,
                    rpm=1000,tpm=450000),
            AIModel(name="Claude-Sonnet-4-H", model="claude-sonnet-4-20250514",reasoning='high',
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=64000,
                    input_price=3.0, cached_price=0.3, cache_write_price=3.75, output_price=15.0,
                    rpm=1000,tpm=450000),
            AIModel(name="Claude-Sonnet-4.5", model="claude-sonnet-4-5-20250929",
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=64000,
                    input_price=3.0, cached_price=0.3, cache_write_price=3.75, output_price=15.0,
                    rpm=1000,tpm=450000),                    
            AIModel(name="Claude-Opus-4", model="claude-opus-4-20250514",
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=32000,
                    input_price=15.0, cached_price=1.5, cache_write_price=18.75, output_price=75.0,
                    rpm=1000,tpm=450000),
            AIModel(name="Claude-Opus-4.1", model="claude-opus-4-1-20250805",
                    max_input_tokens=200000, input_strategy='summarize', max_output_tokens=32000,
                    input_price=15.0, cached_price=1.5, cache_write_price=18.75, output_price=75.0,
                    rpm=1000,tpm=450000),
        ])

@dataclass
class AnthropicConfig(LLMBaseConfig):
    model: str = MODELS.default().model

def convert_exception(ex: Exception) -> LLMError:
    if isinstance(ex, RateLimitError):
        aex: RateLimitError = ex # type: ignore
        return LLMRateLimitError( aex.message or "rate limit error", retry_after=15)
    return LLMError(f"Anthropic API error: {ex}", ex)

class CarrierBotAnthropic(LLMBase):
    """Anthropic LLM を用いるボット（LLMBase 継承）。"""

    def __init__(
        self,
        store,
        match_id: str,
        *,
        name: str|None = None,
        config: Optional[AnthropicConfig] = None,
    ) -> None:
        ai_model = MODELS.get_model(name=name,config=config)
        assert ai_model is not None, f"Unknown Anthropic model: {name}"
        config = config or AnthropicConfig()
        config.model = ai_model.model
        super().__init__(store=store, match_id=match_id, ai_model=ai_model, config=config)
        # サブクラス固有
        self._config: AnthropicConfig = config
        self._client: Anthropic|None = None

    @staticmethod
    def get_model_names() -> AIProvider:
        return MODELS

    # --- Anthropic依存部 ---
    def _ensure_client_ready(self) -> bool:
        if self._client is not None:
            return True
        if USE_ANTHROPIC is None:
            return False
        if not self._config.api_key and not exists_env("ANTHROPIC_API_KEY"):
            return False
        try:
            self._client = Anthropic(
                api_key=self._config.api_key,
                timeout=httpx.Timeout(120.0, read=60.0, write=10.0, connect=2.0),
                )
            return True
        except Exception:
            self._client = None
            return False

    def build_input_params(self,msgs: list[dict[str, str]], *, output_format: type[BaseModel]|None=None) -> tuple[list[TextBlockParam],list[MessageParam],ThinkingConfigEnabledParam|ThinkingConfigDisabledParam]:

        system_prompt:str|NotGiven = NotGiven()
        if msgs[0].get('role')==ROLE_SYSTEM:
            system_prompt = msgs[0].get('content') or NotGiven()
            msgs = msgs[1:]

        if output_format and issubclass(output_format, BaseModel):
            output_prompt = f"Output format: Strictly follow the JSON schema below.\n{output_format.model_json_schema()}"
            if system_prompt is NotGiven or not system_prompt:
                system_prompt = output_prompt
            else:
                system_prompt = system_prompt + "\n\n" + output_prompt

        system_blocks:list[TextBlockParam] = [{
            'type': 'text',
            'text': system_prompt or "",
            'cache_control': {'type': 'ephemeral'}
        }]

        xmsgs: list[MessageParam] = [ ]
        for m in msgs:
            role = m.get('role')
            role = 'assistant' if role==ROLE_AI else 'user'
            content = m.get('content')
            if content:
                xmsgs.append( MessageParam(role=role,content=content ) )

        if len(xmsgs)>0 and xmsgs[-1].get('role')=='user':
            last_msg: MessageParam = xmsgs[-1]
            last_content:list[TextBlockParam] = [{
                'type': 'text',
                'text': str(last_msg.get('content') or ""),
                'cache_control': {'type':'ephemeral'}
            }]
            last_msg['content'] = last_content

        if output_format and issubclass(output_format, BaseModel):
            xmsgs.append( MessageParam(
                role='user',
                content='JSONスキーマに従って回答してください。'
            ) )

        if self.aimodel.reasoning == 'low':
            thinking = ThinkingConfigEnabledParam({ 'type': 'enabled', 'budget_tokens': 2000 })
        elif self.aimodel.reasoning == 'medium':
            thinking = ThinkingConfigEnabledParam({ 'type': 'enabled', 'budget_tokens': 5000 })
        elif self.aimodel.reasoning == 'high':
            thinking = ThinkingConfigEnabledParam({ 'type': 'enabled', 'budget_tokens': 100000 })
        else:
            thinking = ThinkingConfigDisabledParam({ 'type': 'disabled' })

        return system_blocks,xmsgs,thinking

    def count_tokens(self, msgs: list[dict[str, str]], *, output_format: type[BaseModel]|None=None) -> int:
        """メッセージのトークン数を数える。"""
        if not msgs:
            return 0

        # Anthropicのcount_tokensエンドポイントは実際に送信するペイロードと同じ構造を要求する。
        # LLM()と同じロジックでsystem/messagesを組み立て、APIに問い合わせる。

        system_blocks,xmsgs,thinking = self.build_input_params(msgs,output_format=output_format)

        try:
            response = self._client.messages.count_tokens(
                model=self._config.model,
                messages=xmsgs,
                system=system_blocks,
                thinking=thinking,
            ) if self._client is not None else None

            if response is not None:
                input_tokens = getattr(response, 'input_tokens', None)
                if input_tokens is None and isinstance(response, dict):
                    input_tokens = response.get('input_tokens')
                if input_tokens is not None:
                    return int(input_tokens)
        except Exception:
            pass
        return 0

    def LLM(self, msgs:list[dict[str,str]], *, output_format: type[BaseModel]|None=None) -> str:
        if self._client is None:
            raise RuntimeError("Anthropic client not initialized")
        if len(msgs)==0:
            raise RuntimeError("no messages to review")

        temperature = self._config.temperature or 0.2 if self.aimodel.reasoning is None else 1.0

        system_blocks,xmsgs,thinking = self.build_input_params(msgs,output_format=output_format)

        predict_tokens = self.count_tokens(msgs, output_format=output_format)
        ntry = 0
        while True:
            t = self._rating_info.tokens+predict_tokens
            if self.aimodel.tpm and t >= self.aimodel.tpm:
                print(f"Anthropic token limit: wait... {t} / {self.aimodel.tpm}")
                time.sleep(10.0)
                continue
            elif self.aimodel.rpm and self._rating_info.count >= self.aimodel.rpm:
                print(f"Anthropic rate limit: wait... {self._rating_info.count} / {self.aimodel.rpm}")
                time.sleep(10.0)
                continue

            try:
                ntry += 1
                resp: Message = self._client.messages.create(
                    model = self._config.model,
                    max_tokens = self.get_max_output_tokens(),
                    thinking = thinking,
                    system = system_blocks,
                    messages = xmsgs,
                    temperature = temperature,
                )
                break
            except Exception as ex:
                llmerror = convert_exception(ex)
                if isinstance(llmerror, LLMRateLimitError):
                    if ntry < 6:
                        wait = 15
                        print(f"Anthropic rate limit: {llmerror}. wait {wait} sec and retry...")
                        time.sleep(wait)
                        continue
                raise llmerror
        
#anthropic.RateLimitError: Error code: 429 - {'type': 'error', 'error': {'type': 'rate_limit_error', 'message': "This request would exceed your organization's (8d863577-e8f4-4b16-99cc-a6128a02233b) maximum usage increase rate for input tokens per minute. Please scale up your input tokens usage more gradually to stay within the acceleration limit. For details, refer to: https://docs.claude.com/en/api/rate-limits."}, 'request_id': 'req_011CTYQ4npuYDQt9AyEKCGfk'}

        assistant_text: str = ""
        if isinstance(resp, Message) and resp.content and len(resp.content)>0:
            txtblk:ContentBlock = resp.content[0]
            if isinstance(txtblk, TextBlock):
                assistant_text = txtblk.text # type: ignore

        try:
            usage_data: Usage|None = resp.usage if isinstance(resp, Message) and hasattr(resp,'usage') else None
            if usage_data and isinstance(usage_data, Usage):
                input_tokens = usage_data.input_tokens or 0
                output_tokens = usage_data.output_tokens or 0
                cache_read = usage_data.cache_read_input_tokens or 0
                cache_write = usage_data.cache_creation_input_tokens or 0
                if cache_read>0 or cache_write>0:
                    # キャッシュ利用分を入力トークンから減算
                    print(f"Anthropic usage: input={input_tokens} output={output_tokens} cache_read={cache_read} cache_write={cache_write}")
                usage = LLMTokenUsage(
                    prompt=input_tokens + cache_read + cache_write,
                    completion=output_tokens,
                    reasoning=0,
                    cache_read= cache_read,
                    cache_write= cache_write,
                )
                self._set_last_usage_tokens(usage)
        finally:
            # self._set_last_usage_tokens(None)
            pass

        if not assistant_text:
            raise RuntimeError("empty response from model")

        return assistant_text

    def _diag_model_name(self) -> str:  # type: ignore[override]
        return self._config.model
