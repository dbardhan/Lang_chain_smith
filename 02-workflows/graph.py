from langgraph.graph import StateGraph,START,END    
from IPython.display import Image, display
from typing import Annotated,List,TypedDict
from langgraph.graph import add_messages

#1. Define our state
class SimpleState(TypedDict):
    messages: Annotated[List,add_messages]
#2. Create our nodes (agents)
my_graph = StateGraph(SimpleState)
#3. Link nodes with edges
def say_hello(state: SimpleState):
    """Executes Hello part """
    print("Executing say hello")
    return {"messages":["Hello"]}
def say_world(state: SimpleState):
    """Executes world part """
    print("Executing say world")
    return {"messages":["World"]}
my_graph.add_node("hello_node",say_hello)
my_graph.add_node("world_node",say_world)
my_graph.add_edge(START,"hello_node")
my_graph.add_edge("hello_node","world_node")
my_graph.add_edge("world_node",END)
#4. Compile graph
agent = my_graph.compile()
#5. Run the graph
initial_stage = {"message":[]}
final_state = agent.invoke(initial_stage)
print ("===Final output===")
print (final_state)
#display(Image(agent.get_graph().draw_mermaid_png()))