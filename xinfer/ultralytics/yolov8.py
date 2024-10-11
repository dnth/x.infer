from typing import Dict, List

from ultralytics import YOLO

from ..base_model import BaseModel


class YOLOv8(BaseModel):
    def __init__(self, model_name: str = "yolov8n.pt", **kwargs):
        self.model_name = model_name
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.model = YOLO(self.model_name)

    def inference(self, images: str | List[str], **kwargs) -> List[List[Dict]]:
        results = self.model.predict(images, **kwargs)
        batch_results = []
        for result in results:
            coco_format_results = []
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                coco_format_results.append(
                    {
                        "bbox": [x1, y1, width, height],
                        "category_id": int(box.cls),
                        "score": float(box.conf),
                        "class_name": result.names[int(box.cls)],
                    }
                )
            batch_results.append(coco_format_results)
        return batch_results
