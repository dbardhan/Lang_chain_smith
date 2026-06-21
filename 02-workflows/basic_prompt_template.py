from langchain_core.prompts import PromptTemplate

string_prompt = PromptTemplate.from_template("your are a role {role}, answer the question {question} ")

prompt = string_prompt.invoke({"role": "doctor", "question"
                               : "what is the best way to treat a cold?"})
print(prompt)