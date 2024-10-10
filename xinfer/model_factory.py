from rich.console import Console
from rich.table import Table

from .model_registry import ModelRegistry
from .transformers.blip2 import BLIP2, VLRMBlip2


def register_models():
    ModelRegistry.register("transformers", "Salesforce/blip2-opt-2.7b", BLIP2)
    ModelRegistry.register(
        "transformers", "sashakunitsyn/vlrm-blip2-opt-2.7b", VLRMBlip2
    )


def get_model(model_type: str, implementation: str, **kwargs):
    return ModelRegistry.get_model(model_type, implementation, **kwargs)


def list_models():
    console = Console()
    table = Table(title="Available Models")
    table.add_column("Implementation", style="cyan")
    table.add_column("Model Type", style="magenta")

    for model in ModelRegistry.list_models():
        table.add_row(model["implementation"], model["model_type"])

    console.print(table)
