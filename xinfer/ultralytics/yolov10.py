from ..model_registry import register_model
from ..types import ModelInputOutput
from .ultralytics_model import UltralyticsModel


@register_model("ultralytics/yolov10n", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("ultralytics/yolov10s", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("ultralytics/yolov10l", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("ultralytics/yolov10m", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("ultralytics/yolov10x", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
class YOLOv10(UltralyticsModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
