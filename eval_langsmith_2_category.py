from dotenv import load_dotenv
load_dotenv(".env", override=True)

from langsmith import Client
from my_app import classify_book
client = Client()

def my_llm_call(inputs: dict) -> dict:
    answer = classify_book(inputs["title"], inputs["description"])

    return { "answer": answer }

def my_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    if reference_outputs["category"] ==  outputs["answer"]:
        return 1
    else:
        return 0

experiment_results = client.evaluate(
    my_llm_call,
    data="Book Dataset 107",
    evaluators=[my_evaluator],
    experiment_prefix="my-test-eval",
    max_concurrency=10,
)