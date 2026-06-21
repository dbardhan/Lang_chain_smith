from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.context import tracing_v2_enabled
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import time
from dotenv import load_dotenv
load_dotenv()

tracer = LangChainTracer(project_name="llm-evaluator")


llm_answer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_judge = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt for answering
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

# Prompt for judging
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an impartial evaluator. Compare the model's answer to the golden reference."),
    ("user", "Question: {question}\nModel Answer: {answer}\nGolden Reference: {expected}\n\n"
             "Decide if the answer is correct. Respond with 'Correct' or 'Incorrect' and explain briefly.")
])

# Chains
answer_chain = answer_prompt | llm_answer
judge_chain = judge_prompt | llm_judge

test_cases = [
    {"question": "What is the capital of France?", "expected": "Paris"},
    {"question": "2+2=?", "expected": "4"},
    {"question": "Who wrote Hamlet?", "expected": "Shakespeare"},
]

results = []

with tracing_v2_enabled(project_name=tracer.project_name):
    for case in test_cases:
        start = time.time()
        answer = answer_chain.invoke({"question": case["question"]})
        end = time.time()

        # Judge correctness
        verdict = judge_chain.invoke({
            "question": case["question"],
            "answer": answer.content,
            "expected": case["expected"]
        })

        results.append({
            "question": case["question"],
            "expected": case["expected"],
            "answer": answer.content,
            "latency": round(end - start, 3),
            "verdict": verdict.content
        })

# Report
print("\n=== Evaluation Results ===")
for r in results:
    print(f"Q: {r['question']}")
    print(f"Expected: {r['expected']} | Got: {r['answer']}")
    print(f"Latency: {r['latency']}s | Verdict: {r['verdict']}")
    print("-" * 50)

