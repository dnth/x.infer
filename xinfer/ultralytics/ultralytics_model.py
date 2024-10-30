from typing import Dict, List

import torch
from ultralytics import YOLO

from ..models import BaseModel, track_inference


class UltralyticsModel(BaseModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        super().__init__(model_id, device, dtype)
        self.model_type = 'classification' if 'cls' in model_id else 'detection'
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        if self.model_type == 'classification':
            self.model = YOLO(self.model_id, task='classification', **kwargs)
        else:
            self.model = YOLO(self.model_id, **kwargs)

    @track_inference
    def infer_batch(self, images: str | List[str], **kwargs) -> List[List[Dict]]:
        half = self.dtype == torch.float16
        results = self.model.predict(images, device=self.device, half=half, **kwargs)
        batch_results = []
        for result in results:
            
            if self.model_type == 'classification':
                classification_results = []
                probs = result.probs
                classification_results.append({
                    "class_id": int(probs.top1),
                    "score": float(probs.top1conf.cpu().numpy()),
                    "class_name": result.names[int(probs.top1)],
                })
                batch_results.append(classification_results)
                
            else:
                detection_results = []
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    width = x2 - x1
                    height = y2 - y1
                    detection_results.append({
                        "bbox": [x1, y1, width, height],
                        "category_id": int(box.cls),
                        "score": float(box.conf),
                        "class_name": result.names[int(box.cls)],
                    })
                batch_results.append(detection_results)
                
        return batch_results

    @track_inference
    def infer(self, image: str, **kwargs) -> List[List[Dict]]:
        results = self.infer_batch([image], **kwargs)
        return results[0]
