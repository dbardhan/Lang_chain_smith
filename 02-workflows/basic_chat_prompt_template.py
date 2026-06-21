from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

my_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to French."),
    ("user", "{text_to_translate}")
])
llm = ChatOpenAI(model="gpt-4o")
my_prompt_reduced= my_prompt.invoke({"text_to_translate": "Hello, how are you?"})
respose=llm.invoke(my_prompt_reduced)
print(respose.content)