import os

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import requests
import csv
import io
from braintrust import init_dataset


dataset = init_dataset(project="Course-202504", name="books-107")

# Download and parse the CSV data
url = "https://ihower.tw/data/books-dataset-107.csv"
response = requests.get(url)
response.raise_for_status()  # Raise an exception for bad status codes

csv_data = response.text
csv_file = io.StringIO(csv_data)
reader = csv.DictReader(csv_file)

for row in reader:

    id = dataset.insert( input={"title": row['title'], "description": row['description']}, 
                         expected={"category": row['category']}, 
                         metadata={"task": "book_category"})
    
print(dataset.summarize())
