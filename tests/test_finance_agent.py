import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
from langchain.tools import ToolRuntime
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langsmith import tracing_context


class FakeToolCallingModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.fixture
def finance_agent(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "02-workflows" / "finance_agent.py"
    spec = spec_from_file_location("finance_agent", path)
    module = module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)

    with monkeypatch.context() as import_patch:
        import_patch.setattr("dotenv.load_dotenv", lambda: False)
        import_patch.setattr(
            "langchain.chat_models.init_chat_model",
            lambda *args, **kwargs: FakeToolCallingModel(responses=[]),
        )
        spec.loader.exec_module(module)

    return module


@pytest.fixture
def runtime(finance_agent):
    return ToolRuntime(
        state={},
        context=finance_agent.UserContext("user_001", "Alice Johnson", "basic"),
        config={},
        stream_writer=lambda _: None,
        tool_call_id=None,
        store=None,
    )


@pytest.mark.parametrize(
    ("user_id", "currency", "expected_balance"),
    [("user_001", "USD", "2500.00 USD"), ("user_002", "EUR", "1320.00 EUR")],
)
def test_account_balance_uses_the_current_user_and_currency(
    finance_agent, runtime, user_id, currency, expected_balance
):
    runtime.context.user_id = user_id
    runtime.context.preferred_currency = currency

    result = finance_agent.get_account_balance.invoke(
        {"runtime": runtime, "account_type": "Checking"}
    )

    assert result == f"Your Checking account balance is {expected_balance}."


def test_recent_transactions_respect_the_limit(finance_agent, runtime):
    result = finance_agent.get_recent_transactions.invoke(
        {"runtime": runtime, "account_type": "checking", "limit": 2}
    )

    assert result.splitlines() == [
        "Here are your recent transactions for your checking account:",
        "2025-01-15: Grocery Store - -85.50 USD",
        "2025-01-14: Direct Deposit - 3200.00 USD",
    ]


def test_transfer_updates_both_balances(finance_agent, runtime):
    result = finance_agent.transfer_money.invoke(
        {"runtime": runtime, "from_account": "checking", "to_account": "savings", "amount": 125}
    )

    assert result == "Successfully transferred 125.00 USD from your checking account to your savings account."
    accounts = finance_agent.USER_DATABASE[runtime.context.user_id]["accounts"]
    assert accounts["checking"] == 2375
    assert accounts["savings"] == 15125


@pytest.mark.parametrize(
    ("destination", "amount", "message"),
    [("savings", 2501, "Insufficient funds"), ("missing", 125, "not found")],
)
def test_invalid_transfer_keeps_balances_intact(finance_agent, runtime, destination, amount, message):
    accounts = finance_agent.USER_DATABASE[runtime.context.user_id]["accounts"]
    original_balances = accounts.copy()

    with pytest.raises(ValueError, match=message):
        finance_agent.transfer_money.invoke(
            {"runtime": runtime, "from_account": "checking", "to_account": destination, "amount": amount}
        )

    assert accounts == original_balances


@pytest.mark.parametrize(
    ("tier", "expected_greeting"),
    [
        ("basic", "As a basic member"),
        ("premium", "As a premium member"),
        ("VIP", "As a VIP member"),
        ("unknown", "Welcome back to your finance dashboard."),
    ],
)
def test_greeting_uses_membership_tier(finance_agent, runtime, tier, expected_greeting):
    runtime.context.membership_tier = tier

    result = finance_agent.get_personalize_greetings.invoke({"runtime": runtime})

    assert runtime.context.name in result
    assert expected_greeting in result


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Month name prefixes do not match ISO dates")
def test_budget_totals_include_only_the_requested_month(finance_agent, runtime):
    result = finance_agent.calculate_budget.invoke({"runtime": runtime, "month": "January 2025"})

    assert result.splitlines() == [
        "Budget Summary for January 2025:",
        "Total Income: 3337.50 USD",
        "Total Expenses: -1305.50 USD",
        "Net Savings: 2032.00 USD",
    ]


def test_agent_returns_structured_response_after_tool_call(finance_agent, runtime):
    finance_agent.basic_model.responses = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "get_account_balance",
                "args": {"account_type": "checking"},
                "id": "call_balance",
            }],
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "FinancialResponse",
                "args": {"summary": "Balance retrieved", "details": "2500.00 USD"},
                "id": "call_response",
            }],
        ),
    ]

    with tracing_context(enabled=False):
        result = finance_agent.agent.invoke(
            {"messages": [HumanMessage(content="What is my checking balance?")]},
            context=runtime.context,
        )

    tool_results = {
        message.tool_call_id: message.content
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }
    assert tool_results["call_balance"] == "Your checking account balance is 2500.00 USD."
    assert result["structured_response"] == finance_agent.FinancialResponse(
        summary="Balance retrieved", details="2500.00 USD"
    )
