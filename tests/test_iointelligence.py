import sys,os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from openai import OpenAI
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema, JSONSchema
from openai.types.shared_params.response_format_json_object import ResponseFormatJSONObject

class OutputFormat(BaseModel):
    text: str = Field(..., description="The text output from the model.")
    metadata: dict = Field(..., description="Additional metadata about the response.")

def test_run( *, model:str, api_key:str|None=None, output_model: type[BaseModel]|None=None, base_url:str|None=None ):
    print()
    print(f"Running test for model: {model}")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp_fmt = None
        if output_model is not None:
            resp_fmt = ResponseFormatJSONSchema(type="json_schema", json_schema={
                "name": output_model.__name__,
                "schema": output_model.model_json_schema(),
                "strict": False
            })

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            response_format=resp_fmt # type: ignore
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error during test_run: {e}", file=sys.stderr)

def main():
    load_dotenv("config.env")

    test_run( model="gpt-5-nano",output_model=OutputFormat)
    model = "openai/gpt-oss-20b"
    base_url = "https://api.intelligence.io.solutions/api/v1/"
    test_run(
        model=model,
        api_key=os.getenv("IOINTELLIGENCE_API_KEY"),
        base_url=base_url,
    )

    test_run(
        model=model,
        api_key=os.getenv("IOINTELLIGENCE_API_KEY"),
        base_url=base_url,
        output_model=OutputFormat
    )

if __name__ == "__main__":
    main()