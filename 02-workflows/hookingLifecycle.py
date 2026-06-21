from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tracers.schemas import Run
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)  
prompt = ChatPromptTemplate.from_template("give me a short ,simple, fact about {topic}")
fact_chain = prompt | model | StrOutputParser()
def my_listener_on_start(run: Run):
    """""Listener function to be called when a chain run starts"""
    print(f"Run started with ID: {run.id} and name: {run.name}")
    print(f"Input to the run: {run.inputs}")
    print(f"Run started at: {run.start_time}")
    print("parent run id: ", run.parent_run_id )
    print("tags: ", run.tags)
    print("metadata: ", run.extra.get("metadata", {}))

def my_listener_on_end(run: Run):
    """""Listener function to be called when a chain run ends"""
    print(f"Run ended with ID: {run.id} and name: {run.name}")
    print(f"Output of the run: {run.outputs}")
    print(f"Run started at: {run.start_time}")
    print("parent run id: ", run.parent_run_id )
    print("tags: ", run.tags)
    print("metadata: ", run.extra.get("metadata", {}))

fact_chain_listner = fact_chain.with_listeners(on_start=my_listener_on_start,
                                                on_end=my_listener_on_end)    
# model.add_listener("start", my_listener_on_start)
# model.add_listener("end", my_listener_on_end)
response = fact_chain_listner.invoke({"topic": "space"})
print(f"Fact about space: {response}")