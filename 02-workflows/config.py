from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_core.tracers.schemas import Run
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = ChatPromptTemplate.from_template("Tell me the fact about the  {topic}")
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
fact_chain = prompt | model | StrOutputParser()
def my_listener_on_start(run: Run):
    """""Listener function to be called when a chain run starts"""
    print(f"Run started with ID: {run.id} and name: {run.name}")
    print("trace_id", run.trace_id)
    print(f"Input to the run: {run.inputs}")
    print(f"Run started at: {run.start_time}")
    print("parent run id: ", run.parent_run_id )
    print("tags: ", run.tags)
    print("metadata: ", run.extra.get("metadata", {}))
chain_with_listener = fact_chain.with_listeners(on_start=my_listener_on_start)

my_runnable_config = RunnableConfig(
    run_name="fact_chain_run",
    tags=["fact_chain", "example"],
    metadata={"user_id": "user_123", 
    "description": "A chain that provides facts about a given topic.",
    "input_topic_type": "history"}
)

def my_listener_on_end(run: Run):
    """""Listener function to be called when a chain run ends"""
    print(f"Run ended with ID: {run.id} and name: {run.name}")
    print(f"Output of the run: {run.outputs}")
    print(f"Run started at: {run.start_time}")
    print("parent run id: ", run.parent_run_id )
    print("tags: ", run.tags)
    print("metadata: ", run.extra.get("metadata", {}))

print("-----demo 1----")
#Demo of setting configuration with each llm call 
result= chain_with_listener.invoke({"topic": "NetaJi Subhash Chandra bose"},
                                    config=my_runnable_config)
print(f"Fact about NetaJi Subhash Chandra bose: {result}")
print("demo 2===persistent")

my_persistent_config = RunnableConfig(
    run_name="fact_chain_run_persistent",
    tags=["fact_chain_persist", "example2"],
    metadata={"user_id": "user_123", 
    "description": "A chain that provides facts about a given topic.",
    "input_topic_type": "space"}
)

persist_chain_with_listner = chain_with_listener.with_config(my_persistent_config)
persistent_result = persist_chain_with_listner.invoke({"topic":"what is James web telescope"})
print(persistent_result)