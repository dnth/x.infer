from rich.console import Console
from rich.table import Table

from .model_registry import InputOutput, ModelRegistry
from .timm.list_models import download_model_list
from .timm.timm_model import TimmModel
from .transformers.blip2 import BLIP2, VLRMBlip2
from .transformers.moondream import Moondream
from .ultralytics.ultralytics_model import UltralyticsYoloModel


def register_models():
    ModelRegistry.register(
        "transformers",
        "Salesforce/blip2-opt-2.7b",
        BLIP2,
        input_output=InputOutput.IMAGE_TEXT_TO_TEXT,
    )
    ModelRegistry.register(
        "transformers",
        "sashakunitsyn/vlrm-blip2-opt-2.7b",
        VLRMBlip2,
        input_output=InputOutput.IMAGE_TEXT_TO_TEXT,
    )

    ModelRegistry.register(
        "transformers",
        "vikhyatk/moondream2",
        Moondream,
        input_output=InputOutput.IMAGE_TEXT_TO_TEXT,
    )

    ultralytics_models = [
        "yolov8n",
        "yolov8s",
        "yolov8l",
        "yolov8m",
        "yolov8x",
        "yolo11n",
        "yolo11s",
        "yolo11m",
        "yolo11l",
        "yolo11x",
    ]
    for model in ultralytics_models:
        ModelRegistry.register(
            "ultralytics",
            model,
            UltralyticsYoloModel,
            input_output=InputOutput.IMAGE_TO_OBJECTS,
        )

    timm_models = download_model_list()
    for model in timm_models:
        ModelRegistry.register(
            "timm",
            model,
            TimmModel,
            input_output=InputOutput.IMAGE_TO_CLASS,
        )


def create_model(model_id: str, backend: str, **kwargs):
    if backend == "timm":
        kwargs["model_name"] = model_id
    if backend == "ultralytics":
        kwargs["model_name"] = model_id + ".pt"
    return ModelRegistry.get_model(model_id, backend, **kwargs)


def list_models(wildcard: str = None):
    console = Console()
    table = Table(title="Available Models")
    table.add_column("Backend", style="cyan")
    table.add_column("Model ID", style="magenta")
    table.add_column("Input --> Output", style="green")

    for model in ModelRegistry.list_models():
        if wildcard is None or wildcard.lower() in model["model_id"].lower():
            table.add_row(
                model["backend"],
                model["model_id"],
                model["input_output"].value,
            )

    console.print(table)
