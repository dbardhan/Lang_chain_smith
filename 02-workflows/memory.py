from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
#from langchain_community.cache import SQLiteCache

load_dotenv()

class TokenCounterCallback(BaseCallbackHandler):
    """Callback handler to track token usage"""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    
    def on_llm_end(self, response, **kwargs):
        """Called when LLM run ends"""
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)
            self.total_tokens = self.prompt_tokens + self.completion_tokens

# Create callback instance
token_counter = TokenCounterCallback()

# Create model with callback
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    callbacks=[token_counter]
)
set_llm_cache(InMemoryCache()) ##in memory cache
#set_llm_cache(SQLiteCache(database_path="sqlite_cache.db")) ##sqlite cache

# Invoke the model
response = model.invoke([HumanMessage(content="What is Python?")])

# Print results
print(f"Response: {response.content}")
print(f"Prompt tokens: {token_counter.prompt_tokens}")
print(f"Completion tokens: {token_counter.completion_tokens}")
print(f"Total tokens: {token_counter.total_tokens}")

response = model.invoke([HumanMessage(content="What is Python?")])
print(f"Prompt tokens: {token_counter.prompt_tokens}")
print(f"Completion tokens: {token_counter.completion_tokens}")
print(f"Total tokens: {token_counter.total_tokens}")
print("Note: Token counts unchanged because response came from cache no llm has been called. So it is giving the last result.")