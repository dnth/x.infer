import timm

from ..model_registry import register_model
from ..types import ModelInputOutput
from .timm_model import TimmModel


def load_timm_models():
    model_list = timm.list_models("*1k*", pretrained=True)

    for model_id in model_list:
        register_model(
            f"timm/{model_id}", "timm", ModelInputOutput.IMAGE_TO_CATEGORIES
        )(TimmModel)


load_timm_models()
