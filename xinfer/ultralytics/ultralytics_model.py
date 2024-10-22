from typing import Dict, List

import torch
from ultralytics import YOLO

from ..models import BaseModel


class UltralyticsModel(BaseModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError("dtype must be one of 'float32', 'float16', or 'bfloat16'")
        dtype = dtype_map[dtype]

        super().__init__(model_id, device, dtype)
        self.load_model()

    def load_model(self):
        self.model = YOLO(self.model_id)

    def infer_batch(self, images: str | List[str], **kwargs) -> List[List[Dict]]:
        with self.track_inference_time():
            half = self.dtype == torch.float16
            results = self.model.predict(
                images, device=self.device, half=half, **kwargs
            )
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
        self.update_inference_count(len(batch_results))
        return batch_results

    def infer(self, image: str, **kwargs) -> List[List[Dict]]:
        with self.track_inference_time():
            results = self.infer_batch([image], **kwargs)
        return results[0]
