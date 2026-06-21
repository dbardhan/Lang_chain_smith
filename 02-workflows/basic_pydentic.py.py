from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
class President(BaseModel):
    name: str = Field(description="The name of the president")
    passport_number: Optional[str] = Field(description="The passport number of the president")
    country: str = Field(description="The country of the president")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that provides information about presidents."),
    ("user", "Who is the president of {country}?")
])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(President)
prompt_reduced = prompt.format(country="France")
response = structured_llm.invoke(prompt_reduced)
print(response)