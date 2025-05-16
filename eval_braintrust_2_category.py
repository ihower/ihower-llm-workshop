import os

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from braintrust import Eval, init_dataset

from my_app import classify_book

def my_llm_call(input: dict) -> dict:
    answer = classify_book(input["title"], input["description"])
    return { "answer": answer }

def exact_match(input, expected, output):
    if expected["category"] == output["answer"]:
        return 1
    else:
        return 0

Eval(
    "Course-202504", # project name
    task=my_llm_call,
    data=init_dataset(project="Course-202504", name="books-107"),
    scores=[exact_match]
)