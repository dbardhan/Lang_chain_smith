from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
president_schema = {
    "name": "president_schema",
    "description": "A schema to get the president of a country",
    "parameters": {
        "type": "object",
        "properties": {
            "country": {
                "type": "string",
                "description": "The country to get the president for",
            },
            "name": {
                "type": "string",
                "description": "The name of the president",
            },
            "passport_number": {
                "type": ["string", "null"],
                "description": "The passport number of the president",
            },
        }
    },
    "required": ["country", "name"]
}
structured_llm = model.with_structured_output(president_schema)
response = structured_llm.invoke("Who is the president of France?")
print(response)
