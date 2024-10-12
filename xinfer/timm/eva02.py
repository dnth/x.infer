from ..model_registry import ModelInputOutput, register_model
from .timm_model import TimmModel


@register_model(
    "eva02_tiny_patch14_336.mim_in22k_ft_in1k", "timm", ModelInputOutput.IMAGE_TO_CLASS
)
@register_model(
    "eva02_small_patch14_336.mim_in22k_ft_in1k", "timm", ModelInputOutput.IMAGE_TO_CLASS
)
@register_model(
    "eva02_base_patch14_448.mim_in22k_ft_in1k", "timm", ModelInputOutput.IMAGE_TO_CLASS
)
@register_model(
    "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
    "timm",
    ModelInputOutput.IMAGE_TO_CLASS,
)
@register_model(
    "eva02_large_patch14_448.mim_in22k_ft_in1k",
    "timm",
    ModelInputOutput.IMAGE_TO_CLASS,
)
@register_model(
    "eva02_large_patch14_448.mim_in22k_ft_in22k_in1k",
    "timm",
    ModelInputOutput.IMAGE_TO_CLASS,
)
@register_model(
    "eva02_large_patch14_448.mim_m38m_ft_in1k",
    "timm",
    ModelInputOutput.IMAGE_TO_CLASS,
)
@register_model(
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    "timm",
    ModelInputOutput.IMAGE_TO_CLASS,
)
class EVA02(TimmModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
