import sys,os
from pydantic import BaseModel, Field


if __name__=="__main__":
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from server.services.match import MatchStore
from server.services.ai_llm_base import LLMBase, ROLE_SYSTEM, ROLE_USER, ROLE_AI
from server.services.ai_openai import CarrierBotOpenAI
from server.services.ai_anthropic import CarrierBotAnthropic
from server.services.ai_gemini import CarrierBotGemini

def test_json_fmt(model: type[BaseModel]):
    if hasattr(model, "json_fmt"):
        print(model.json_fmt()) # type: ignore
    else:
        print("json_fmt関数がありません")
        print(model.model_json_schema())

def test():

    class SampleModel(BaseModel):
        name: str
        value: int

    class Model2(BaseModel):
        question: str
        answer: str

    class Model3(BaseModel):
        question: str
        thought: str
        answer: str

    query_list = [
        ("Hello, who won the world series in 2020?",SampleModel),
        ("そのとき、猫の毛並みがどうなったか、しってますか？",Model2),
        ("お好み焼ききん太の営業時間は？ クーポンはいつまで使える？",None)
    ]

    print("test ResponseModel")
    store = MatchStore()

    for m in range(3):
        bot:LLMBase|None = None
        if m==0:
            bot = None # CarrierBotOpenAI(store, match_id="test")
        elif m==1:
            bot = None # CarrierBotAnthropic(store, match_id="test")
        elif m==2:
            bot = CarrierBotGemini(store, match_id="test")

        if bot is None or not bot._ensure_client_ready():
            print(" client not ready")
            continue

        mesgs:list[dict[str, str]] = [
            {"role": ROLE_SYSTEM, "content": "You are a helpful assistant.きん太のクーポン有効期間は、9/12から10/13までです。"},
        ]

        for q,mdl in query_list:

            mesgs.append({"role": ROLE_USER, "content": q})
            tokens_predict = bot.count_tokens(mesgs, output_format=mdl)
            response = bot.LLM(mesgs, output_format=mdl)
            print(response)
            if mdl:
                resdata = mdl.model_validate_json(response)
                print(resdata)

            prompt_tokens = bot._last_token_usage.prompt_tokens
            diff = tokens_predict - prompt_tokens
            print(f"predict count: {tokens_predict}")
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Diff: {diff}")

            mesgs.append({"role": ROLE_AI, "content": response})

if __name__=="__main__":
    test()