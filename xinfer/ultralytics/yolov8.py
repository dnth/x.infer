from ..model_registry import ModelInputOutput, register_model
from .ultralytics_model import UltralyticsModel


@register_model("yolov8n", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("yolov8s", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("yolov8l", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("yolov8m", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("yolov8x", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
class YOLOv8(UltralyticsModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
