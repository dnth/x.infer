from ..model_registry import ModelInputOutput, register_model
from .ultralytics_model import UltralyticsModel


@register_model("yolov10n", "ultralytics", ModelInputOutput.IMAGE_TO_OBJECTS)
@register_model("yolov10s", "ultralytics", ModelInputOutput.IMAGE_TO_OBJECTS)
@register_model("yolov10l", "ultralytics", ModelInputOutput.IMAGE_TO_OBJECTS)
@register_model("yolov10m", "ultralytics", ModelInputOutput.IMAGE_TO_OBJECTS)
@register_model("yolov10x", "ultralytics", ModelInputOutput.IMAGE_TO_OBJECTS)
class YOLOv10(UltralyticsModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
