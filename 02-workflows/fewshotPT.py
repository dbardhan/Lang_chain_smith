from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate ,FewShotPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
prompt_formatter = PromptTemplate.from_template("{input} is a programming language.\
                                   It is used for web development, data analysis,\
                                   artificial intelligence, and more. It is known for \
                                   its simplicity and readability."
                                   )
fewshot_prompt_formatter = FewShotPromptTemplate(  
    examples=[
        {"input": "Python"},
        {"input": "JavaScript"},
        {"input": "Java"},
    ],
    
    example_prompt=prompt_formatter,
    #prefix="Here are some programming languages and their key features:",
    suffix="What are some of the key features of {input}?",
    input_variables=["input"],
    
)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# Invoke the model with few-shot prompt
resolved_prompt = fewshot_prompt_formatter.invoke({"input": "C++"})
print (f"Resolved Prompt: {resolved_prompt}")
response = model.invoke(resolved_prompt)
print(f"Response: {response.content}")