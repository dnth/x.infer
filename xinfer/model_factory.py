from rich.console import Console
from rich.table import Table

from .model_registry import InputOutput, ModelRegistry
from .transformers.blip2 import BLIP2, VLRMBlip2
from .transformers.moondream import Moondream


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


def create_model(model_id: str, backend: str, **kwargs):
    return ModelRegistry.get_model(model_id, backend, **kwargs)


# TODO: list by backend or wildcard
def list_models():
    console = Console()
    table = Table(title="Available Models")
    table.add_column("Backend", style="cyan")
    table.add_column("Model ID", style="magenta")
    table.add_column("Input --> Output", style="green")

    for model in ModelRegistry.list_models():
        table.add_row(
            model["backend"],
            model["model_id"],
            model["input_output"].value,
        )

    console.print(table)
