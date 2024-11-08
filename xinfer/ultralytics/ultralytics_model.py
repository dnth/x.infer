import os

import torch

from ultralytics import YOLO

from ..models import BaseModel, track_inference
from ..types import Box, Category, Result


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
            16: "Right Ankle",
        }

    def load_model(self, **kwargs):
        self.model = YOLO(self.model_id, **kwargs)

    @track_inference
    def infer_batch(self, images: list[str], **kwargs) -> list[Result]:
        use_half_precision = self.dtype in [torch.float16, torch.bfloat16]
        self.results = self.model.predict(
            images, device=self.device, half=use_half_precision, **kwargs
        )
        batch_results = []

        for result in self.results:
            if "cls" in self.model_id:
                classification_results = []

                top5_classes_idx = result.probs.top5
                top5_classes_scores = result.probs.top5conf

                for class_idx, score in zip(top5_classes_idx, top5_classes_scores):
                    classification_results.append(
                        Category(score=float(score), label=result.names[class_idx])
                    )

                batch_results.append(Result(categories=classification_results))

            elif "pose" in self.model_id:
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
                            detection_keypoints.append(
                                {
                                    "point": point.tolist(),
                                    "score": float(score),
                                    "name": self.pose_name.get(i, f"Keypoint {i}"),
                                }
                            )
                        keypoints_results.append(detection_keypoints)
                    batch_results.append(keypoints_results)
                else:
                    batch_results.append([])

            elif "seg" in self.model_id:
                segmentation_results = []
                masks = result.masks
                boxes = result.boxes
                classes = boxes.cls.cpu().numpy().astype(int).tolist()
                scores = boxes.conf.cpu().numpy().tolist()
                names = [result.names[c] for c in classes]
                if masks is not None:
                    for i in range(len(masks)):
                        segmentation_results.append(
                            {
                                "mask": masks.data[i].cpu().numpy(),
                                "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                                "category_id": classes[i],
                                "score": scores[i],
                                "class_name": names[i],
                            }
                        )
                    batch_results.append(segmentation_results)
                else:
                    batch_results.append([])

            elif "yolo" in self.model_id:
                detection_results = []
                boxes = result.boxes
                for box in boxes:
                    detection_results.append(
                        Box(
                            x1=float(box.xyxy[0][0].cpu().numpy()),
                            y1=float(box.xyxy[0][1].cpu().numpy()),
                            x2=float(box.xyxy[0][2].cpu().numpy()),
                            y2=float(box.xyxy[0][3].cpu().numpy()),
                            score=float(box.conf.cpu().numpy()),
                            label=result.names[int(box.cls.cpu().numpy())],
                        )
                    )
                batch_results.append(Result(boxes=detection_results))

            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}")

        return batch_results

    @track_inference
    def infer(self, image: str, **kwargs) -> list[dict]:
        results = self.infer_batch([image], **kwargs)
        return results[0]

    def render(self, save_path: str = "./", **kwargs):
        for _, r in enumerate(self.results):
            # save results to disk
            file_name = os.path.basename(r.path)
            file_name = os.path.join(save_path, file_name)
            r.save(filename=f"{file_name}")
            print(f"Saved Render Imgae to {file_name}")
