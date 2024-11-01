from ..model_registry import ModelInputOutput, register_model
from .ultralytics_model import UltralyticsModel


@register_model("ultralytics/yolov11n", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("ultralytics/yolov11s", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("ultralytics/yolov11m", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model("ultralytics/yolov11l", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES)
@register_model(
    "ultralytics/yolov11n-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES
)
@register_model(
    "ultralytics/yolov11s-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES
)
@register_model(
    "ultralytics/yolov11m-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES
)
@register_model(
    "ultralytics/yolov11l-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES
)
class YOLOv11(UltralyticsModel):
    def __init__(self, model_id: str, **kwargs):
        model_id = model_id.replace("v", "")
        super().__init__(model_id, **kwargs)
