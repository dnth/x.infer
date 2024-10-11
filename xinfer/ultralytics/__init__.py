from .ultralytics_model import UltralyticsYoloModel

ultralytics_models = [
    "yolov8n",
    "yolov8s",
    "yolov8l",
    "yolov8m",
    "yolov8x",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10l",
    "yolov10x",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo11l",
    "yolo11x",
]


__all__ = ["UltralyticsYoloModel", "ultralytics_models"]
