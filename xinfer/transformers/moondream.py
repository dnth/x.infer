import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model_registry import ModelInputOutput, register_model
from ..models import BaseModel


@register_model(
    "vikhyatk/moondream2", "transformers", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
class Moondream(BaseModel):
    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str = "2024-08-26",
        device: str = "cpu",
        dtype: str = "float32",
    ):
        super().__init__(model_id, device, dtype)
        self.revision = revision
        self.load_model()

    def preprocess(
        self,
        images: str | list[str],
    ):
        if not isinstance(images, list):
            images = [images]

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

        return processed_images

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, revision=self.revision
        ).to(self.device, self.dtype)

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def infer(self, image: str, prompt: str = None, **generate_kwargs):
        with self.track_inference_time():
            image = self.preprocess(image)
            encoded_image = self.model.encode_image(image)
            output = self.model.answer_question(
                question=prompt,
                image_embeds=encoded_image,
                tokenizer=self.tokenizer,
                **generate_kwargs,
            )

        self.update_inference_count(1)
        return output

    def infer_batch(self, images: list[str], prompts: list[str], **generate_kwargs):
        with self.track_inference_time():
            images = self.preprocess(images)
            prompts = [prompt for prompt in prompts]

            outputs = self.model.batch_answer(
                images, prompts, self.tokenizer, **generate_kwargs
            )

        self.update_inference_count(len(images))
        return outputs
