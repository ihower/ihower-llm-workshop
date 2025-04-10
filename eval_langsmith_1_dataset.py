import requests
import csv
import io
from langsmith import Client

client = Client()

# Programmatically create a dataset in LangSmith
# For other dataset creation methods, see:
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application
dataset = client.create_dataset(
    dataset_name="Book Dataset 107", description="A book dataset with 107 entries"
)

# Download and parse the CSV data
url = "https://ihower.tw/data/books-dataset-107.csv"
response = requests.get(url)
response.raise_for_status()  # Raise an exception for bad status codes

csv_data = response.text
csv_file = io.StringIO(csv_data)
reader = csv.DictReader(csv_file)

# Create examples from CSV data
examples = []
for row in reader:
    examples.append(
        {
            "inputs": {"title": row['title'], "description": row['description']},
            "outputs": {"category": row['category']}
        }
    )


# Add examples to the dataset
client.create_examples(dataset_id=dataset.id, examples=examples)

print(f"Dataset '{dataset.name}' created with {len(examples)} examples.")