from dataclasses import dataclass
from enum import Enum


class ModelInputOutput(Enum):
    IMAGE_TO_TEXT = "image --> text"
    IMAGE_TEXT_TO_TEXT = "image-text --> text"
    TEXT_TO_TEXT = "text --> text"
    IMAGE_TO_BOXES = "image --> boxes"
    IMAGE_TO_CATEGORIES = "image --> categories"
    IMAGE_TO_MASKS = "image --> masks"
    IMAGE_TO_POINTS = "image --> points"


@dataclass
class ModelInfo:
    id: str
    implementation: str
    input_output: ModelInputOutput


@dataclass
class Category:
    score: float
    label: str


@dataclass
class Result:
    # For image classification models
    categories: list[Category] = None

    # For image-text to text models
    text: str = None
