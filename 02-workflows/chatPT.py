from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI 


example_formatter = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

example_set = [
    {"input": "What is the solution of 5 m_m 3?", "output": "The solution is 15"},
    {"input": "What is the solution of 7 m_m 2?", "output": "The solution is 14"},
    {"input": "What is the solution of 9 m_m 4?", "output": "The solution is 36"},
    {"input": "What is the solution of 6 m_m 5?", "output": "The solution is 30"},
]
few_shot_template = FewShotChatMessagePromptTemplate(
    example_prompt=example_formatter,
    examples=example_set,)


main_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert mathematician and a helpful assistant."),
    few_shot_template,
    ("human", "{user_final_prompt}"),
])
invoked_prompt = main_prompt.invoke({
    "user_final_prompt": "what is the solution of 8 m_m 9?"})
print(f"Invoked Prompt: {invoked_prompt.to_string()}")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke(main_prompt.invoke({
    "user_final_prompt": "Please compute 8 m_m 9. I dont know what is the operation of m_m?",
}))
print(response.content)
