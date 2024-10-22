from ..model_registry import ModelInputOutput, register_model
from .vision2seq import Vision2SeqModel


@register_model(
    "Salesforce/blip2-opt-2.7b", "transformers", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
@register_model(
    "Salesforce/blip2-opt-6.7b", "transformers", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
@register_model(
    "Salesforce/blip2-flan-t5-xxl", "transformers", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
@register_model(
    "Salesforce/blip2-opt-6.7b-coco",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class BLIP2(Vision2SeqModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
