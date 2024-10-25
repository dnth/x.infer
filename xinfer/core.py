from loguru import logger
from rich.console import Console
from rich.table import Table

from .model_registry import model_registry
from .utils import TimmModel, UltralyticsModel, Vision2SeqModel


def create_model(model: str | TimmModel | Vision2SeqModel | UltralyticsModel, **kwargs):
    """
    Create a model instance.

    Parameters
    ----------
    model : str | TimmModel | Vision2SeqModel | UltralyticsModel
        The model to create.
        TIMM, Vision2Seq, and Ultralytics models type here is to support user passing in the models directly.
        This is useful for models not registered in the model registry.

        Eg:
        ```python
        model = UltralyticsModel("yolov5n6u")
        model = xinfer.create_model(model)
        ```
    """
    if isinstance(model, (TimmModel, Vision2SeqModel, UltralyticsModel)):
        return model
    return model_registry.get_model(model, **kwargs)


def list_models(search: str = None, limit: int = 20, interactive: bool = False):
    import pandas as pd

    rows = []
    for model_info in model_registry.list_models():
        if search is None or search.lower() in model_info.id.lower():
            rows.append(
                {
                    "Implementation": model_info.implementation,
                    "Model ID": model_info.id,
                    "Input --> Output": model_info.input_output.value,
                }
            )

    if not rows:
        logger.warning(
            "No models found matching the criteria.\n"
            "Perhaps install the relevant dependencies? For example, `pip install xinfer[timm]`"
        )
        return

    if interactive:
        from itables import init_notebook_mode

        logger.info(
            "Showing interactive table in Jupyter Notebook. Type in the search bar to filter the models."
        )

        init_notebook_mode(all_interactive=True)
        return pd.DataFrame(rows)

    if len(rows) > limit:
        rows = rows[:limit]
        rows.append(
            {"Implementation": "...", "Model ID": "...", "Input --> Output": "..."}
        )
        rows.append(
            {"Implementation": "...", "Model ID": "...", "Input --> Output": "..."}
        )

        logger.info(
            f"Showing only top {limit} models. Change the `limit` parameter to see more."
        )

    console = Console()
    table = Table(title="Available Models")
    table.add_column("Implementation", style="cyan")
    table.add_column("Model ID", style="magenta")
    table.add_column("Input --> Output", style="green")

    for row in rows:
        table.add_row(row["Implementation"], row["Model ID"], row["Input --> Output"])

    console.print(table)
