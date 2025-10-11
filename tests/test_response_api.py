import unittest
import json

from pydantic import BaseModel, Field

from openai import OpenAI
from openai.types.responses.response_prompt_param import ResponsePromptParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam
from openai.types.chat.chat_completion_developer_message_param import ChatCompletionDeveloperMessageParam
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from openai.types.responses.response_format_text_config_param import ResponseFormatTextConfigParam
from openai.types.responses.response_format_text_json_schema_config_param import ResponseFormatTextJSONSchemaConfigParam
from openai.lib._parsing._completions import type_to_response_format_param

from dotenv import load_dotenv

class TestSample(BaseModel):
    date: str = Field(..., description="current date")
    time: str = Field(..., description="current time")


def to_content(resp):
    # 応答テキスト抽出（いくつかの SDK 表現に頑健に対応）
    content = None
    try:
        # まず output_text があれば利用
        content = getattr(resp, "output_text", None)
        if not content:
            out = getattr(resp, "output", None)
            # resp.output がリスト/オブジェクトなら最初の content -> text を参照
            if out:
                first = out[0] if isinstance(out, (list, tuple)) and len(out) > 0 else out
                c = None
                if isinstance(first, dict):
                    c = first.get("content")
                else:
                    c = getattr(first, "content", None)
                if isinstance(c, list) and len(c) > 0:
                    part = c[0]
                    if isinstance(part, dict):
                        content = part.get("text") or part.get("markdown") or json.dumps(part)
                    else:
                        content = str(part)
                elif isinstance(c, str):
                    content = c
    except Exception:
        content = None
    return content

def get_usage(resp):
    usage = None
    try:
        u = getattr(resp, "usage", None)
        if u:
            prompt_tokens = getattr(u, "prompt_tokens", 0) or 0
            completion_tokens = getattr(u, "completion_tokens", 0) or 0
            total_tokens = getattr(u, "total_tokens", 0) or 0
            usage = {
                "prompt": int(prompt_tokens),
                "completion": int(completion_tokens),
                "total": int(total_tokens),
            }
    except Exception:
        usage = None
    return usage

def test_structured_output():

    client = OpenAI()
    mdl="gpt-5-nano"
    resp = client.responses.parse(
        model=mdl,
        input="いま何時？",
        text_format=TestSample
    )
    # print(resp)
    cid = resp.id
    print("Response ID:", cid)
    tt = to_content(resp)
    print("Response:", tt)

def test():

    client = OpenAI()
    mdl="gpt-5-nano"
    prompt = "白猫は三匹います。"
    query1 = "白猫は何匹いますか？"
    input2= "黒猫は1匹います。"
    query2 = "黒猫は何匹いますか？"
    input3= "三毛猫は五匹います。"
    query3 = "三毛猫は何匹いますか？"
    #input1 = "たぬきは昼寝、狐は水浴びをしています。黒猫は何匹いますか？"
    print("----------")
    last_input = input2 + " " + query1
    resp = client.responses.create(
        model=mdl,
        previous_response_id=None,
        instructions=prompt,
        input=last_input,
    )
    # print(resp)
    cid1 = resp.id
    print("#最初の質問")
    print("Query:", last_input)
    print("Response ID:", cid1)
    tt = to_content(resp)
    print("Response:", tt)

    # print("----------")
    last_input = input3 + " " + query2
    # resp = client.responses.create(
    #     model=mdl,
    #     previous_response_id=None,
    #     instructions=prompt,
    #     input=last_input,
    # )
    # # print(resp)
    # cid2a = resp.id
    # print("#1回目の情報を引き継がずに質問(答えられないはず)")
    # print("Query:", last_input)
    # print("Response ID:", cid2a)
    # tt = to_content(resp)
    # print("Response:", tt)

    print("----------")
    resp = client.responses.create(
        model=mdl,
        previous_response_id=cid1,
        instructions=prompt,
        input=last_input,
    )
    # print(resp)
    cid2 = resp.id
    print("#1回目の情報を引き継いで質問(答えられるはず)")
    print("Query:", last_input)
    print("Response ID:", cid2)
    tt = to_content(resp)
    print("Response:", tt)


    print("----------")
    last_input = query3
    resp = client.responses.create(
        model=mdl,
        previous_response_id=cid1,
        instructions=prompt,
        input=last_input,
    )
    # print(resp)
    cid3a = resp.id
    print("#1回目の情報を引き継いで質問(答えられないはず)")
    print("Query:", last_input)
    print("Response ID:", cid3a)
    tt = to_content(resp)
    print("Response:", tt)

    print("----------")
    resp = client.responses.create(
        model=mdl,
        previous_response_id=cid2,
        instructions=prompt,
        input=last_input,
    )
    # print(resp)
    cid3b = resp.id
    print("#1回目の情報を引き継いで質問(これはどうなるか？)")
    print("Query:", last_input)
    print("Response ID:", cid3b)
    tt = to_content(resp)
    print("Response:", tt)

    print("----------")
    print("----------")



if __name__ == "__main__":
    load_dotenv("config.env")
    load_dotenv("../config.env")
    #test_structured_output()
    test()