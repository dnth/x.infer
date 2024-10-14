import time

import requests
import torch
from loguru import logger
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from ..base_model import BaseModel


class Vision2SeqModel(BaseModel):
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs).to(
            self.device, torch.bfloat16
        )

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()

    def preprocess(
        self,
        images: str | Image.Image | list[str | Image.Image],
        prompts: str | list[str],
    ):
        if not isinstance(images, list):
            images = [images]
        if not isinstance(prompts, list):
            prompts = [prompts]

        if len(images) != len(prompts):
            raise ValueError("The number of images and prompts must be the same")

        processed_images = []
        for image in images:
            if isinstance(image, str):
                if image.startswith(("http://", "https://")):
                    image = Image.open(requests.get(image, stream=True).raw).convert(
                        "RGB"
                    )
                else:
                    raise ValueError("Input string must be an image URL")
            elif not isinstance(image, Image.Image):
                raise ValueError("Input must be either an image URL or a PIL Image")
            processed_images.append(image)

        return self.processor(
            images=processed_images, text=prompts, return_tensors="pt"
        ).to(self.device, torch.bfloat16)

    def predict(self, preprocessed_input, **generate_kwargs):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=torch.bfloat16
        ):
            return self.model.generate(**preprocessed_input, **generate_kwargs)

    def postprocess(self, predictions):
        outputs = self.processor.batch_decode(predictions, skip_special_tokens=True)
        return [output.replace("\n", "").strip() for output in outputs]

    def infer(self, image, prompt, verbose=False, **generate_kwargs):
        start_time = time.perf_counter()
        preprocessed_input = self.preprocess(image, prompt)
        prediction = self.predict(preprocessed_input, **generate_kwargs)
        result = self.postprocess(prediction)[0]
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        if verbose:
            logger.info(f"Inference time: {inference_time*1000:.4f} ms")
        return result

    def infer_batch(self, images, prompts, verbose=False, **generate_kwargs):
        start_time = time.perf_counter()
        preprocessed_input = self.preprocess(images, prompts)
        predictions = self.predict(preprocessed_input, **generate_kwargs)
        results = self.postprocess(predictions)
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        if verbose:
            logger.info(f"Inference time: {inference_time*1000:.4f} ms")
        return results
