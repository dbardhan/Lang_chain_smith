from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()
my_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that translates English to French."),
    ("user", "{text_to_translate}")
])
llm = ChatOpenAI(model="gpt-4o")
my_prompt_reduced1= my_prompt.invoke({"text_to_translate": "Hello, how are you?"})
my_prompt_reduced2= my_prompt.invoke({"text_to_translate": "Thanks you, Have nice day"})

respose=llm.batch([my_prompt_reduced1, my_prompt_reduced2]  )
for r in respose:
    print(r.content)
