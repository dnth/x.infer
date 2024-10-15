from loguru import logger
from rich.console import Console
from rich.table import Table

from .model_registry import model_registry
from .utils import TimmModel, UltralyticsModel, Vision2SeqModel


def create_model(model: str | TimmModel | Vision2SeqModel | UltralyticsModel, **kwargs):
    if isinstance(model, (TimmModel, Vision2SeqModel, UltralyticsModel)):
        return model
    return model_registry.get_model(model, **kwargs)


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

    if not rows:
        logger.warning(
            "No models found matching the criteria.\n"
            "Perhaps install the relevant dependencies? For example, `pip install xinfer[timm]`"
        )
        return

    if len(rows) > limit:
        rows = rows[:limit]
        rows.append(("...", "...", "..."))
        rows.append(("...", "...", "..."))

    for row in rows:
        table.add_row(*row)

    console.print(table)
