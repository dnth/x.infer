import csv
from io import StringIO
from typing import List

import requests


def download_model_list() -> List[str]:
    # URL of the CSV file
    url = "https://raw.githubusercontent.com/huggingface/pytorch-image-models/6ee638a0953aad999e1ec87310c9543ee5c4aa9e/results/results-imagenet.csv"

    # Download the CSV file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad responses

    # Parse the CSV content
    csv_content = StringIO(response.text)
    reader = csv.DictReader(csv_content)

    # Extract model names from the 'model' column
    model_list = [row["model"] for row in reader if "model" in row]

    return model_list
