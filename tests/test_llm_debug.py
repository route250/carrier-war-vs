import sys,os
if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from server.services.ai_base import AIStat
from server.services.match import MatchStore
from server.schemas import MatchCreateRequest, MatchJoinRequest
from server.services.ai_openai import OpenAIConfig, CarrierBotOpenAI
from server.services.ai_anthropic import AnthropicConfig, CarrierBotAnthropic
from server.services.ai_gemini import GeminiConfig, CarrierBotGemini
from server.services.ai_cpu import Config, CarrierBotMedium
from server.services.ai_llm_base import LLMBase

from pathlib import Path
from dotenv import load_dotenv

class LlmTestRun:

    def setUp(self):
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

        self.store = MatchStore()

    def bot_setup(self, provider: str, model: str) -> LLMBase|None:

        # PvPでマッチ作成（A側に人間スロット作成）
        resp = self.store.create(MatchCreateRequest(mode="pvp", config=None, display_name="HUMAN"))
        match_id = resp.match_id
        match = self.store._matches[match_id]

        # LLMボットをB側に参加（スレッドは起動しない）
        if provider == 'openai':
            config = OpenAIConfig(model=model)
            bot = CarrierBotOpenAI(store=self.store, match_id=match_id, name=model, config=config)
        elif provider == 'anthropic':
            config = AnthropicConfig(model=model)
            bot = CarrierBotAnthropic(store=self.store, match_id=match_id, name=model,config=config)
        elif provider == 'gemini':
            config = GeminiConfig(model=model)
            bot = CarrierBotGemini(store=self.store, match_id=match_id, name=model, config=config)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if not bot._ensure_client_ready():
            print(f"Skipping {provider} LLM bot test because API key is not set or client cannot be initialized.")
            return
        bot.stat = AIStat.READY
        if not bot.startup():
            print(f"Skipping {provider} LLM bot test because startup() failed.")
            return
        # B側にJoinしてトークン/サイドを確定
        join = self.store.join(match_id, MatchJoinRequest(display_name=bot.name))
        bot.token = join.player_token
        bot.side = join.side

        # 初期位置・HPの控え
        self.b0 = match.map.get_carrier_by_side("B")
        self.a0 = match.map.get_carrier_by_side("A")
        assert self.b0 is not None and self.a0 is not None
        self.b_init = (self.b0.pos.x, self.b0.pos.y)
        return bot

    def _ai_step(self, bot: LLMBase, turn:int):
        match = self.store._matches[bot.match_id]
        # mapを含むスナップショットを渡す（ai_geminiはpayload.mapを使用可）
        payload = self.store.snapshot(bot.match_id, bot.token)
        if turn==1:
            payload.map = self.store.get_map_array(bot.match_id)
        orders = bot.think(payload)
        assert bot.submit_orders(orders) == True, "LLMボットの命令提出に失敗"
        # もう片側は未提出のため、テスト用に直接ターン解決
        match._resolve_turn_minimal()
        match.side_a.orders = None
        match.side_b.orders = None

    def _run_llm(self, provider, model, nturns:int):
        bot = self.bot_setup(provider, model)
        if bot is None:
            print(f"\nSkipping test for {provider} model {model} due to setup failure.")
            return

        print(f"\n\nTesting LLM bot {bot.name} for launch and carrier move...")
        match = self.store._matches[bot.match_id]
        launched = False
        moved = False
        try:
            for t in range(1,nturns+1):
                self._ai_step(bot, t)
                # B側の編隊が発艦しているか
                for u in match.map.get_squadrons_by_side("B"):
                    if u.state != 'onboard' and u.state != "lost":
                        launched = True
                        print(f"[{bot.name} 編隊を発艦しました")
                # B空母がどこかで動いたか
                b_now = match.map.get_carrier_by_side("B")
                if b_now and (b_now.pos.x, b_now.pos.y) != self.b_init:
                    moved = True
                if moved:
                    print(f"[{bot.name} 空母が移動しました")
                    #return True
        finally:
            pass
            #bot.cleanup()

        if not launched:
            print("LLMボットが発艦（攻撃/索敵行動）しませんでした")
        if not moved:
            print("LLMボットの空母が移動しませんでした")

        if not launched:
            print(f"[{bot.name} 編隊を発艦しませんでした")
            xx = bot.debug_call("\n".join([
                "プロンプトを改善するためのデバッグ情報をください。あなたの出力に関しての質問です。",
                "1. あなたは発艦指示を出しましたか？",
                "3. 発艦指示を出した理由、出さなかった理由を教えて下さい。"
                "2. 実際には艦載機が発艦していません。つまりlaunch_targetは全ての応答でnullでした。これはなぜでしょうか？プロンプトの内容が誤解を与えていたら訂正したいと思います。",
                "出力は、プレーンテキストで、箇条書きで、簡潔に答えてください。"
            ]))
            print(f"Debug call response:\n{xx}")

        return False

def main():
    provider = "anthropic"
    model = "Claude-Haiku-3"

    run = LlmTestRun()
    run.setUp()
    run._run_llm(provider, model, 10)


if __name__ == '__main__':
    main()
