# import csv
# from io import StringIO
# from typing import List

# import requests

from .eva02 import EVA02
# from .timm_model import TimmModel

# def download_model_list() -> List[str]:
#     url = "https://raw.githubusercontent.com/huggingface/pytorch-image-models/6ee638a0953aad999e1ec87310c9543ee5c4aa9e/results/results-imagenet.csv"
#     response = requests.get(url)
#     response.raise_for_status()  # Raise an exception for bad responses

#     csv_content = StringIO(response.text)
#     reader = csv.DictReader(csv_content)
#     model_list = [row["model"] for row in reader if "model" in row]

#     return model_list


# timm_models = download_model_list()


# __all__ = ["TimmModel", "timm_models"]
