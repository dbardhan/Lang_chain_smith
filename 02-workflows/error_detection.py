from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.schemas import Run
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
load_dotenv()

def failing_function(dinput_dict: Dict[str, Any]) -> str:
    """A function that raises an error for demonstration purposes."""
    topic = dinput_dict.get("topic", "unknown")
    if "error" in topic.lower():

        raise ValueError("This is a simulated error for testing error detection.")
    elif "json" in topic.lower():
        # Simulate a JSON parsing error
        malformed_json = '{"name": "John Doe", "age": 30'  # Missing closing brace
        data = json.loads(malformed_json)  # This will raise a JSONDecodeError
        raise ValueError(f"Parsed data: {data}")
    elif "network" in topic.lower() :
        # Simulate a network error
        raise ConnectionError("Simulated network error: Unable to connect to the server.")
    else:
        return f"Successfully processed topic: {topic}"
    
error_ruannable = RunnableLambda(failing_function)
def error_listener_on_error(run: Run, error: Exception):
    """Listener function to be called when an error occurs during a run."""
    print(f"Error detected in run with ID: {run.id} and name: {run.name}")
    print(f"Error type: {type(error).__name__}")
    print(f"Error message: {str(error)}")
    print(f"Run inputs at the time of error: {run.inputs}")
    print(f"Run outputs at the time of error: {run.outputs}")
    print(f"Run started at: {run.start_time}")
    print("parent run id: ", run.parent_run_id )
    print("tags: ", run.tags)
    print("metadata: ", run.extra.get("metadata", {}))
    print("-" * 50)
    print("error details")
    print(run.error)

error_ruannable_with_listener = error_ruannable.with_listeners(on_error=error_listener_on_error)

try :
    # This will raise a ValueError and trigger the error listener
    result = error_ruannable_with_listener.invoke({"topic": "error somewhere here"})
    print(f"Result: {result}")
except Exception as e:
    print(f"Caught exception: {e}")