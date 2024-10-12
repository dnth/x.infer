from io import BytesIO
from typing import Dict, List

import requests
import timm
import torch
from loguru import logger
from PIL import Image

from ..base_model import BaseModel
from .imagenet1k_classes import IMAGENET2012_CLASSES


class TimmModel(BaseModel):
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        logger.info(f"Loading model: {self.model_id}")
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        logger.info(f"Loading model: {self.model_id}")
        self.model = timm.create_model(self.model_id, pretrained=True, **kwargs).eval()

    def inference(self, image_url: str, top_k: int = 5) -> List[List[Dict]]:
        logger.info(f"Running inference on {image_url}")
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        data_config = timm.data.resolve_model_data_config(self.model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        img = transforms(img).unsqueeze(0)

        output = self.model(img)

        topk_probabilities, topk_class_indices = torch.topk(
            output.softmax(dim=1) * 100, k=top_k
        )

        im_classes = list(IMAGENET2012_CLASSES.values())
        class_names = [im_classes[i] for i in topk_class_indices[0]]

        return [
            {"class": class_name, "id": int(class_idx), "confidence": float(prob)}
            for class_name, class_idx, prob in zip(
                class_names, topk_class_indices[0], topk_probabilities[0]
            )
        ]
