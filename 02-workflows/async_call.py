from langchain_openai import ChatOpenAI
import asyncio
from dotenv import load_dotenv
load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)
async def stream_event_handler(event):
    # Process the event data
    event_data = event.data
    print(f"Received event: {event_data}")

    # Generate a response using the LLM
    #response = await model.ainvoke([f"Process this event data: {event_data}"])
    async for response_chunk in model.astream_events([f"Process this event data: {event_data}"]):
        print(f"Generated response: {response_chunk}")
    
print("Starting event stream...")
# Simulate receiving events from a stream
async def simulate_event_stream():
    # Simulate receiving events from a stream
    for i in range(5):
        event_data = f"Event {i+1}"
        await stream_event_handler(type('Event', (object,), {'data': event_data})())
        #await stream_event_handler(Event(event_data))
        await asyncio.sleep(1)  # Simulate a delay between 

""" class Event:
    def __init__(self, data):
        self.data = data
 """
if __name__ == "__main__":
    asyncio.run(simulate_event_stream())



