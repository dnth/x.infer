from ..model_registry import register_model
from ..types import ModelInputOutput
from .ultralytics_model import UltralyticsModel

model_configs = [
    ("ultralytics/yolov11n", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov11s", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov11m", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov11l", "ultralytics", ModelInputOutput.IMAGE_TO_BOXES),
    ("ultralytics/yolov11n-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES),
    ("ultralytics/yolov11s-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES),
    ("ultralytics/yolov11m-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES),
    ("ultralytics/yolov11l-cls", "ultralytics", ModelInputOutput.IMAGE_TO_CATEGORIES),
    ("ultralytics/yolov11n-pose", "ultralytics", ModelInputOutput.IMAGE_TO_POINTS),
    ("ultralytics/yolov11s-pose", "ultralytics", ModelInputOutput.IMAGE_TO_POINTS),
    ("ultralytics/yolov11m-pose", "ultralytics", ModelInputOutput.IMAGE_TO_POINTS),
    ("ultralytics/yolov11l-pose", "ultralytics", ModelInputOutput.IMAGE_TO_POINTS),
    ("ultralytics/yolov11n-seg", "ultralytics", ModelInputOutput.IMAGE_TO_MASKS),
    ("ultralytics/yolov11s-seg", "ultralytics", ModelInputOutput.IMAGE_TO_MASKS),
    ("ultralytics/yolov11m-seg", "ultralytics", ModelInputOutput.IMAGE_TO_MASKS),
    ("ultralytics/yolov11l-seg", "ultralytics", ModelInputOutput.IMAGE_TO_MASKS),
    # add more model refer https://github.com/ultralytics/assets/releases/
]


def batch_register_model(configs):
    def decorator(cls):
        for config in configs:
            register_model(*config)(cls)
        return cls

    return decorator


@batch_register_model(model_configs)
class YOLOv11(UltralyticsModel):
    def __init__(self, model_id: str, **kwargs):
        model_id = model_id.replace("v", "")
        super().__init__(model_id, **kwargs)
