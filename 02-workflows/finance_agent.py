from multiprocessing import get_context
from urllib import response

from langchain.agents import create_agent
from langchain.tools import tool,ToolRuntime
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Literal
from pydantic import BaseModel, Field
load_dotenv()  # Load environment variables from .env file
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (wrap_model_call
                                         ,wrap_tool_call
                                         ,dynamic_prompt
                                         ,ModelRequest
                                         ,ModelResponse)
from langchain_core.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.structured_output import ToolStrategy

USER_DATABASE = {
    "user_001": {
        "name": "Alice Johnson",
        "accounts": {
            "checking": 2500.00,
            "savings": 15000.00,
            "investment": 45000.00,
        },
        "transactions": {
            "checking": [
                {"date": "2025-01-15", "description": "Grocery Store", "amount": -85.50},
                {"date": "2025-01-14", "description": "Direct Deposit", "amount": 3200.00},
                {"date": "2025-01-13", "description": "Electric Bill", "amount": -120.00},
                {"date": "2025-01-12", "description": "Restaurant", "amount": -45.00},
                {"date": "2025-01-10", "description": "Gas Station", "amount": -55.00},
            ],
            "savings": [
                {"date": "2025-01-01", "description": "Interest Payment", "amount": 12.50},
                {"date": "2024-12-15", "description": "Transfer from Checking", "amount": 500.00},
            ],
            "investment": [
                {"date": "2025-01-14", "description": "Dividend - AAPL", "amount": 125.00},
                {"date": "2025-01-10", "description": "Buy - VTI", "amount": -1000.00},
            ],
        },
    },
    "user_002": {
        "name": "Bob Smith",
        "accounts": {
            "checking": 1200.00,
            "savings": 8000.00,
            "investment": 22000.00,
        },
        "transactions": {
            "checking": [
                {"date": "2025-01-15", "description": "Coffee Shop", "amount": -5.50},
                {"date": "2025-01-14", "description": "Freelance Payment", "amount": 1500.00},
            ],
            "savings": [
                {"date": "2025-01-01", "description": "Interest Payment", "amount": 6.50},
            ],
            "investment": [
                {"date": "2025-01-12", "description": "Dividend - VTI", "amount": 45.00},
            ],
        },
    },
}


basic_model = init_chat_model(
    "gpt-4o-mini",
    temperature=0.5,
    max_tokens=512
)

premium_model = init_chat_model(
    "gpt-4o",
    max_tokens=2048
)

vip_model = init_chat_model(
    "gpt-4o"
)


##=====
# middleware to dynamically select the model based on user membership tier
##===
@wrap_tool_call
def gracefull_error_handling(request: ToolCallRequest, handler) -> ToolMessage:
    """
    Middleware to gracefully handle errors during tool execution.
    If an error occurs, it returns a user-friendly message instead of crashing.
    """
    tool_name = request.tool.name

    try:
        return handler(request) 
    except ValueError as e:
        msg = f"Error in tool '{tool_name}': {str(e)}"
        print (f"[ERROR] Caught Value exception in {tool_name}: {str(e)}")  # Log the error for debugging 
        return ToolMessage(tool_call_id=request.tool_call.get("id")
                           #,tool_call_name=tool_name
                           , content=msg)
    except KeyError as e:
        msg = f"Error in tool '{tool_name}': {str(e)}"
        print (f"[ERROR] Caught KeyError exception in {tool_name}: {str(e)}")  # Log the error for debugging
        return ToolMessage(tool_call_id=request.tool_call.get("id")
                           ,tool_call_name=tool_name
                           , content=msg)
    except Exception as e:  
        user_context: UserContext = request.runtime.context
        error_message = f"Oops! Something went wrong while processing your request, {user_context.name}. Please try again later."
        print(f"Error for user {user_context.name}: {str(e)}")  # Log the error for debugging
        return ToolMessage(tool_call_id=request.tool.lc_id
                           ,tool_call_name=tool_name
                           ,content=error_message)

@wrap_model_call
def select_model_based_on_membership(request: ModelRequest,handler) -> ModelResponse:
    """ 
        Selects model based on user's membership tier
    """

    user_context: UserContext = request.runtime.context
    tier = user_context.membership_tier.lower()
    if tier == "basic":
        request.override(model = basic_model)
        print("Using basic model for user:", user_context.name)
    elif tier == "premium":
        request.override(model = premium_model)
        print("Using premium model for user:", user_context.name)
    elif tier == "vip":
        request.override(model = vip_model)
        print("Using VIP model for user:", user_context.name)
    else:
        request.override(model = basic_model)  # default to basic if tier is unknown
        print("Using default basic model for user:", user_context.name)

    return handler(request)
@dynamic_prompt
def tier_based_system_prompt(request: ModelRequest) -> str:
    """
    Generates a system prompt based on the user's membership tier.
    """
    user_context: UserContext = request.runtime.context
    tier = user_context.membership_tier.lower()
    user_name = user_context.name
    base_prompt = f"""You are a personal finance assistant helping {user_name}.

            Your capabilities:
            - Check account balances (checking, savings, investment)
            - View recent transactions
            - Calculate budget recommendations
            - Provide personalized greetings"""
		
    if tier == "premium":
        return base_prompt + """

            PREMIUM MEMBER BENEFITS:
            - Provide helpful explanations with your responses
            - Offer occasional tips for financial improvement
            - Be friendly and informative
            - Balance detail with brevity"""

    elif tier == "vip":
        return base_prompt + """

            VIP MEMBER BENEFITS:
            - Provide detailed, comprehensive financial analysis
            - Offer proactive suggestions for wealth growth
            - Include market insights when relevant
            - Be thorough and consultative in your responses
            - Take extra time to explain complex concepts"""

    else:  # basic
        return base_prompt + """

            Guidelines:
            - Be concise and direct
            - Answer questions efficiently
            - Focus on the specific request
            - Keep responses brief but helpful"""


@dataclass
class UserContext:
    user_id: str
    name: str
    membership_tier: str  #basic, premium, or vip
    preferred_currency: str = "USD"

class FinancialResponse(BaseModel):
    summary: str = Field(description="A brief summary of the response (1 - 2 sentences)")
    details: str = Field(description= "Detailed explanation or data")
    action_items: list[str] = Field(
        default_factory=list,
        description="List of recommended actions the user should take"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any warnings or concerns to highlight"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="high",
        description="Confidence level in the advice provided"
    )

    
@tool 
def transfer_money(runtime: ToolRuntime[UserContext], 
                  from_account: str, 
                  to_account: str, 
                  amount: float) -> str:
    """
    Transfers money between two accounts for the user.
    args:
        runtime (ToolRuntime): The runtime context for the tool.
        from_account (str): The account to transfer money from.
        to_account (str): The account to transfer money to.
        amount (float): The amount of money to transfer.
    """
    user_context: UserContext = runtime.context
    user_id = user_context.user_id
    currency = user_context.preferred_currency

    user_data = USER_DATABASE.get(user_id)
    
    if not user_data:
        raise ValueError(f"User ID {user_id} not found.")
    
    accounts = user_data.get("accounts", {})
    
    if from_account.lower() not in accounts :
        raise ValueError(f"From account '{from_account}' not found for user {user_context.name}.")
    if to_account.lower() not in accounts:
        raise ValueError(f"To account  '{to_account}' not found for user {user_context.name}.")

    if accounts[from_account.lower()] < amount:
        raise ValueError(f"Insufficient funds in {from_account} account for user {user_context.name}.")
    
    # Perform the transfer
    accounts[from_account.lower()] -= amount
    accounts[to_account.lower()] += amount
    return f"Successfully transferred {amount:.2f} {currency} from your {from_account} account to your {to_account} account."

@tool
def get_account_balance(runtime: ToolRuntime[UserContext]
                        , account_type: str) -> str:
    """
    Retrieves the balance for a specific account type (checking, savings, investment) for the user.
    args:
        runtime (ToolRuntime): The runtime context for the tool.
        account_type (str): The type of account to retrieve the balance for.
    """
    user_context: UserContext = runtime.context
    user_id = user_context.user_id
    currency = user_context.preferred_currency

    
    user_data = USER_DATABASE.get(user_id)
    
    if not user_data:
        raise ValueError(f"User ID {user_id} not found.")
    
    accounts = user_data.get("accounts", {})
    balance = accounts.get(account_type.lower())
    
    if balance is None:
        raise ValueError(f"Account type '{account_type}' not found for user {user_context.name}.")
    else:
        if currency != "USD":
            balance =  balance * 1.1  # Example conversion rate for demonstration purposes       
    return f"Your {account_type} account balance is {balance:.2f} {currency}."
@tool
def get_recent_transactions(runtime: ToolRuntime[UserContext]
                            , account_type: str
                            , limit: int = 5) -> str:
    """
    Retrieves recent transactions for a specific account type (checking, savings, investment) for the user.
    args:
        runtime (ToolRuntime): The runtime context for the tool.
        account_type (str): The type of account to retrieve transactions for.
        limit (int): The number of recent transactions to retrieve (default is 5).
    """
    user_context: UserContext = runtime.context
    user_id = user_context.user_id
    currency = user_context.preferred_currency
    user_data = USER_DATABASE.get(user_id)
    
    if not user_data:
        raise ValueError(f"User ID {user_id} not found.")
    
    transactions = user_data.get("transactions", {}).get(
        account_type.lower())
    
    if transactions is None:
        raise ValueError(f"Account type '{account_type}' not found for user {user_context.name}.")
    
    recent_transactions = transactions[:limit]
    
    transaction_list = "\n".join(
        [f"{t['date']}: {t['description']} - {t['amount']:.2f} {currency}" for t in recent_transactions]
    )
    
    return f"Here are your recent transactions for your {account_type} account:\n{transaction_list}"    
@tool
def calculate_budget(runtime: ToolRuntime[UserContext]
                     , month: str) -> str:
    """
    Calculates a simple budget for the user based on their recent transactions for a given month.
    args:
        runtime (ToolRuntime): The runtime context for the tool.
        month (str): The month to calculate the budget for (e.g., "January 2025").
    """
    user_context: UserContext = runtime.context
    user_id = user_context.user_id
    currency = user_context.preferred_currency

    user_data = USER_DATABASE.get(user_id)
    
    if not user_data:
        raise ValueError(f"User ID {user_id} not found.")
    
    transactions = []
    for account_transactions in user_data.get("transactions", {}).values():
        transactions.extend(account_transactions)
    
    # Filter transactions for the specified month
    monthly_transactions = [
        t for t in transactions if t["date"].startswith(month.split()[0][:3])
    ]
    
    total_income = sum(t["amount"] for t in monthly_transactions if t["amount"] > 0)
    total_expenses = sum(t["amount"] for t in monthly_transactions if t["amount"] < 0)
    
    budget_summary = (
        f"Budget Summary for {month}:\n"
        f"Total Income: {total_income:.2f} {currency}\n"
        f"Total Expenses: {total_expenses:.2f} {currency}\n"
        f"Net Savings: {(total_income + total_expenses):.2f} {currency}"
    )
    
    return budget_summary   

@tool
def get_personalize_greetings(runtime: ToolRuntime[UserContext]) -> str:
    """
    Provides a personalized greeting message based on the user's name and membership tier.
    args:
        runtime (ToolRuntime): The runtime context for the tool.
    """
    name = runtime.context.name
    membership_tier = runtime.context.membership_tier
    greetings = {
        "basic": f"Hello {name}! Welcome back to your finance dashboard. As a basic member, you have access to essential financial tools to help you manage your money effectively.",
        "premium": f"Hi {name}! Great to see you again. As a premium member, you have access to advanced financial insights and personalized recommendations to optimize your finances.",
        "vip": f"Welcome {name}! It's wonderful to have you here. As a VIP member, you enjoy exclusive financial services and tailored advice to help you achieve your financial goals with ease.",
    }
    
    return greetings.get(membership_tier.lower(), f"Hello {name}! Welcome back to your finance dashboard.")


SYSTEM_PROMPT = """"
You are a helpful personal finance assistant.

Your capabilities:
- Check account balances (checking, savings, investment)
- View recent transactions
- Calculate budget recommendations
- Provide personalized greetings

Guidelines:
- Be helpful and informative
- Always start by greeting the user 
- Provide clear, actionable advice
- Use tools to get accurate, user-specific information
- Format monetary values clearly
- Tailor advice based on the user's membership tier
"""

agent = create_agent(

    model=basic_model,
    tools=[get_account_balance,
           get_recent_transactions, 
           calculate_budget, 
           get_personalize_greetings,
           transfer_money],

    name="FinanceAgent",
    middleware=[
        select_model_based_on_membership,
        tier_based_system_prompt,
        gracefull_error_handling,
    ],
    
    #system_prompt=SYSTEM_PROMPT,
    context_schema=UserContext,
    response_format=ToolStrategy(FinancialResponse),
    
)
if __name__ == "__main__":
    def main():
        alice_context = UserContext(
            user_id="user_001",
            name="Alice Johnson",
            membership_tier="vip",
            preferred_currency="USD"
        )
        bob_context = UserContext(
            user_id="user_002",
            name="Bob Smith",
            membership_tier="basic",
            preferred_currency="EUR"
        )
        # # Test 1: Personalized greeting for bob
        """
        print("\n👥 Personalized greeting for Bob")
        message = "What is my checking account balance?"
        response = agent.invoke({"message": [{"role": "user", "content": message}]},
                                context= bob_context)
        
        print(f"🤖 Response: {response['messages'][-1].content}")
        """

        # Test 4: Financial situation and advice
        """
        financial_situation_query = "What's my financial situation? Check all my accounts and give me advice"
        print("\n👥 Same query, different treatment")
        print("-"* 40)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": financial_situation_query}]},
            context=alice_context
        )
        print(f" 🤖 Agent: {response['messages'][-1].content}") 
        """
        # Test 6: Error handling - insufficient funds
        """
        insufficient_amount_prompt = "Transfer $5000 from checking to savings"
        print(f"\n📝 Query: '{insufficient_amount_prompt}' (should fail)")
        print("-" * 40)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": insufficient_amount_prompt}]},
            context=alice_context)
        print(f"🤖 Agent: {response['messages'][-1].content}") 
        """
        
        # Test 7: Error handling - same account
        """
        print("\n📝 Query: 'Transfer $100 from checking to checking' (should fail)")
        print("-" * 40)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": "Transfer $100 from checking to checking"}]},
            context=bob_context,
        )
        print(f"🤖 Agent: {response['messages'][-1].content}") 
        """
        # Test 8: Structured Response
        query = "What's my financial situation? Check all my accounts and give me advice"
        print("\n👥 Alice - Financial Breakdown")
        print("-"* 40)
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            context=alice_context
        )
        #print(f"🤖 Agent: {response}") 
        structures_response : FinancialResponse = response.get("structured_response")
        print("\nSTRUCTURED RESPONSE")
        print(f"\n Summary:\n {structures_response.summary}")
        print(f"\n Details:\n {structures_response.details}")
        print("\n✅ Action Items:")
        for item in structures_response.action_items:
            print(f"* {item}")

        print("\n⚠️ Warnings: ")
        for warning in structures_response.warnings:
            print(f"* {warning}")
        print(f"\n✅ Confidence Level: {structures_response.confidence}")

    main()