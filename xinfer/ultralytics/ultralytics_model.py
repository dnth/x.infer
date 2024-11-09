import os

import torch

from ultralytics import YOLO

from ..models import BaseXInferModel, track_inference
from ..types import Box, Category, Mask, Pose, Result

COCO_KEYPOINT_LABELS = [
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
]


class UltralyticsModel(BaseXInferModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

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
                pose_results = []
                pose_keypoints = result.keypoints

                for person_points in pose_keypoints:
                    pose_results.append(
                        Pose(
                            keypoints=person_points.xy.cpu().numpy().tolist(),
                            scores=person_points.conf.cpu().numpy().tolist(),
                            labels=COCO_KEYPOINT_LABELS,
                        )
                    )

                batch_results.append(Result(poses=pose_results))

            elif "seg" in self.model_id:
                segmentation_results = []
                detection_results = []
                masks = result.masks

                boxes = result.boxes
                classes = boxes.cls.cpu().numpy().astype(int).tolist()
                # scores = boxes.conf.cpu().numpy().tolist()
                names = [result.names[c] for c in classes]

                if masks:
                    for i in range(len(masks)):
                        segmentation_results.append(
                            Mask(xy=masks.xy[i].tolist()),
                        )
                        detection_results.append(
                            Box(
                                x1=float(boxes.xyxy[i][0].cpu().numpy()),
                                y1=float(boxes.xyxy[i][1].cpu().numpy()),
                                x2=float(boxes.xyxy[i][2].cpu().numpy()),
                                y2=float(boxes.xyxy[i][3].cpu().numpy()),
                                score=float(boxes.conf[i].cpu().numpy()),
                                label=names[i],
                            ),
                        )

                batch_results.append(
                    Result(masks=segmentation_results, boxes=detection_results)
                )

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
