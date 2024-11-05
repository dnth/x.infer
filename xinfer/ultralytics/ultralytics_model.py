import os
from typing import Dict, List

import torch

from ultralytics import YOLO

from ..models import BaseModel, track_inference


class UltralyticsModel(BaseModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)
        self.pose_name = {
            0: "Nose",
            1: "Left Eye",
            2: "Right Eye",
            3: "Left Ear",
            4: "Right Ear",
            5: "Left Shoulder",
            6: "Right Shoulder",
            7: "Left Elbow",
            8: "Right Elbow",
            9: "Left Wrist",
            10: "Right Wrist",
            11: "Left Hip",
            12: "Right Hip",
            13: "Left Knee",
            14: "Right Knee",
            15: "Left Ankle",
            16: "Right Ankle"
        }

    def load_model(self, **kwargs):
        self.model = YOLO(self.model_id, **kwargs)

    @track_inference
    def infer_batch(self, images: str | List[str], **kwargs) -> List[List[Dict]]:
        half = self.dtype == torch.float16
        self.results = self.model.predict(images, device=self.device, half=half, **kwargs)
        batch_results = []

        for result in self.results:
            
            if 'cls' in self.model_id:
                classification_results = []
                probs = result.probs
                classification_results.append({
                    "class_id": int(probs.top1),
                    "score": float(probs.top1conf.cpu().numpy()),
                    "class_name": result.names[int(probs.top1)],
                })
                batch_results.append(classification_results)

            elif 'pose' in self.model_id:
                keypoints_results = []
                keypoints = result.keypoints
                if keypoints is not None:
                    xy = keypoints.xy.cpu().numpy()
                    conf = keypoints.conf.cpu().numpy()
                    for idx in range(len(xy)):
                        detection_keypoints = []
                        for i in range(len(self.pose_name)):
                            point = xy[idx][i]
                            score = conf[idx][i]
                            detection_keypoints.append({
                                "point": point.tolist(),
                                "score": float(score),
                                "name": self.pose_name.get(i, f"Keypoint {i}"),
                            })
                        keypoints_results.append(detection_keypoints)
                    batch_results.append(keypoints_results)
                else:
                    batch_results.append([])

            elif 'seg' in self.model_id:
                segmentation_results = []
                masks = result.masks
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy().astype(int).tolist()
                scores = boxes.conf.cpu().numpy().tolist()
                names = [result.names[c] for c in classes]
                if masks is not None:
                    for i in range(len(masks)):
                        segmentation_results.append({
                            "mask": masks.data[i].cpu().numpy(),
                            "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                            "category_id": classes[i],
                            "score": scores[i],
                            "class_name": names[i],
                        })
                    batch_results.append(segmentation_results)
                else:
                    batch_results.append([])

            elif 'yolo' in self.model_id:
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

            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

        return batch_results

    @track_inference
    def infer(self, image: str, **kwargs) -> List[Dict]:
        results = self.infer_batch([image], **kwargs)
        return results[0]

    def render(self, save_path: str = './', **kwargs):
        for _, r in enumerate(self.results): 
            # save results to disk
            file_name = os.path.basename(r.path)
            file_name = os.path.join(save_path, file_name)
            r.save(filename=f"{file_name}")
            print (f"Saved Render Imgae to {file_name}")
