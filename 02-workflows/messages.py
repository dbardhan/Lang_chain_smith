from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage,SystemMessage  
from dotenv import load_dotenv
load_dotenv()

my_prompt = HumanMessage(content="Who is president of France?")
my_prompt2 = HumanMessage(content="Who is president of Ukraine?")
my_system_message = SystemMessage(content="Your all response should be in french")
llm = ChatOpenAI(model="gpt-4o")
llm_response1=llm.batch([[my_system_message, my_prompt],
                          [my_system_message, my_prompt2]])

for r in llm_response1:
    print(r.content)    
#print(llm_response1.content)