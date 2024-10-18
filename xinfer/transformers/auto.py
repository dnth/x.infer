import time

import requests
import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from ..models import BaseModel


class Vision2SeqModel(BaseModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        logger.info(f"Model: {model_id}")
        logger.info(f"Device: {device}")
        logger.info(f"Dtype: {dtype}")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError("dtype must be one of 'float32', 'float16', or 'bfloat16'")
        dtype = dtype_map[dtype]

        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs).to(
            self.device, self.dtype
        )

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()

    def preprocess(
        self,
        images: str | list[str],
        prompts: str | list[str],
    ):
        if not isinstance(images, list):
            images = [images]
        if not isinstance(prompts, list):
            prompts = [prompts]

        if len(images) != len(prompts):
            raise ValueError("The number of images and prompts must be the same")

        processed_images = []
        for image_path in images:
            if not isinstance(image_path, str):
                raise ValueError("Input must be a string (local path or URL)")

            if image_path.startswith(("http://", "https://")):
                image = Image.open(requests.get(image_path, stream=True).raw).convert(
                    "RGB"
                )
            else:
                # Assume it's a local path
                try:
                    image = Image.open(image_path).convert("RGB")
                except FileNotFoundError:
                    raise ValueError(f"Local file not found: {image_path}")

            processed_images.append(image)

        return self.processor(
            images=processed_images, text=prompts, return_tensors="pt"
        ).to(self.device, self.dtype)

    def predict(self, preprocessed_input, **generate_kwargs):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=self.dtype
        ):
            return self.model.generate(**preprocessed_input, **generate_kwargs)

    def postprocess(self, predictions):
        outputs = self.processor.batch_decode(predictions, skip_special_tokens=True)
        return [output.replace("\n", "").strip() for output in outputs]

    def infer(self, image, prompt, **generate_kwargs):
        with self.stats.track_inference_time():
            preprocessed_input = self.preprocess(image, prompt)
            prediction = self.predict(preprocessed_input, **generate_kwargs)
            result = self.postprocess(prediction)[0]

        self.stats.update_inference_count(1)
        return result

    def infer_batch(self, images, prompts, **generate_kwargs):
        with self.stats.track_inference_time():
            preprocessed_input = self.preprocess(images, prompts)
            predictions = self.predict(preprocessed_input, **generate_kwargs)
            results = self.postprocess(predictions)

        self.stats.update_inference_count(len(images))
        return results
