from ..model_registry import register_model
from ..types import ModelInputOutput
from .ultralytics_model import UltralyticsModel

model_configs = [
    ("ultralytics/yolov8n", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov8s", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov8m", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov8l", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov8x", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov8n-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES),
    ("ultralytics/yolov8s-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES),
    ("ultralytics/yolov8m-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES),
    ("ultralytics/yolov8l-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES),
    ("ultralytics/yolov8n-pose", "ultralytics", ModelInputOutput.IMAGE_TO_POINTS),
    ("ultralytics/yolov8s-pose", "ultralytics", ModelInputOutput.IMAGE_TO_POINTS),
    ("ultralytics/yolov8m-pose", "ultralytics", ModelInputOutput.IMAGE_TO_POINTS),
    ("ultralytics/yolov8l-pose", "ultralytics", ModelInputOutput.IMAGE_TO_POINTS),
    ("ultralytics/yolov8n-seg", "ultralytics", ModelInputOutput.IMAGE_TO_MASKS),
    ("ultralytics/yolov8s-seg", "ultralytics", ModelInputOutput.IMAGE_TO_MASKS),
    ("ultralytics/yolov8m-seg", "ultralytics", ModelInputOutput.IMAGE_TO_MASKS),
    ("ultralytics/yolov8l-seg", "ultralytics", ModelInputOutput.IMAGE_TO_MASKS),
    # add more model refer https://github.com/ultralytics/assets/releases/
]


def batch_register_model(configs):
    def decorator(cls):
        for config in configs:
            register_model(*config)(cls)
        return cls

    return decorator


@batch_register_model(model_configs)
class YOLOv8(UltralyticsModel):
    def __init__(self, model_id: str, **kwargs):
        super().__init__(model_id, **kwargs)
