from rich.console import Console
from rich.table import Table

from .model_registry import model_registry

# from .timm import TimmModel, timm_models
# from .transformers import BLIP2
# from .transformers.moondream import Moondream
# from .ultralytics import UltralyticsYoloModel, ultralytics_models

# def register_models():
#     ModelRegistry.register(
#         "transformers",
#         "Salesforce/blip2-opt-2.7b",
#         BLIP2,
#         input_output=InputOutput.IMAGE_TEXT_TO_TEXT,
#     )

#     ModelRegistry.register(
#         "custom-transformers",
#         "vikhyatk/moondream2",
#         Moondream,
#         input_output=InputOutput.IMAGE_TEXT_TO_TEXT,
#     )

#     for model in ultralytics_models:
#         ModelRegistry.register(
#             "ultralytics",
#             model,
#             UltralyticsYoloModel,
#             input_output=InputOutput.IMAGE_TO_OBJECTS,
#         )

#     for model in timm_models:
#         ModelRegistry.register(
#             "timm",
#             model,
#             TimmModel,
#             input_output=InputOutput.IMAGE_TO_CLASS,
#         )


# def create_model(model_id: str, implementation: str, **kwargs):
#     if implementation == "timm":
#         kwargs["model_name"] = model_id
#     if implementation == "ultralytics":
#         kwargs["model_name"] = model_id + ".pt"
#     if implementation == "transformers":
#         kwargs["model_name"] = model_id
#     return ModelRegistry.get_model(model_id, implementation, **kwargs)


def create_model(model_id: str, **kwargs):
    return model_registry.get_model(model_id, **kwargs)


def list_models(wildcard: str = None, limit: int = 20):
    console = Console()
    table = Table(title="Available Models")
    table.add_column("Implementation", style="cyan")
    table.add_column("Model ID", style="magenta")
    table.add_column("Input --> Output", style="green")

    rows = []
    for model_info in model_registry.list_models():
        if wildcard is None or wildcard.lower() in model_info.id.lower():
            rows.append(
                (
                    model_info.implementation,
                    model_info.id,
                    model_info.input_output.value,
                )
            )

    if len(rows) > limit:
        rows = rows[:limit]
        rows.append(("...", "...", "..."))
        rows.append(("...", "...", "..."))

    for row in rows:
        table.add_row(*row)

    console.print(table)
