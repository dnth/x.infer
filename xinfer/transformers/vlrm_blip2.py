import torch

from ..model_registry import ModelInputOutput, register_model
from .blip2 import BLIP2


@register_model(
    "sashakunitsyn/vlrm-blip2-opt-2.7b",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class VLRMBlip2(BLIP2):
    def __init__(self, model_name: str = "sashakunitsyn/vlrm-blip2-opt-2.7b", **kwargs):
        super().__init__(model_name, **kwargs)
        self.load_vlrm_weights()

    def load_vlrm_weights(self):
        from huggingface_hub import hf_hub_download

        finetuned_weights_state_dict = torch.load(
            hf_hub_download(
                repo_id="sashakunitsyn/vlrm-blip2-opt-2.7b",
                filename="vlrm-blip2-opt-2.7b.pt",
            )
        )
        self.model.load_state_dict(finetuned_weights_state_dict, strict=False)
