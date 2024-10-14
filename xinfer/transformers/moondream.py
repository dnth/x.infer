import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..base_model import BaseModel
from ..model_registry import ModelInputOutput, register_model


@register_model(
    "vikhyatk/moondream2", "transformers", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
class Moondream(BaseModel):
    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str = "2024-08-26",
        **kwargs,
    ):
        self.model_id = model_id
        self.revision = revision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, revision=self.revision
        )

        if self.device == "cuda":
            self.model = self.model.to(self.device, torch.bfloat16)

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def infer(self, image: str, prompt: str = None, **generate_kwargs):
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                raise ValueError("Input string must be an image URL for BLIP2")
        else:
            raise ValueError(
                "Input must be either an image URL or a PIL Image for BLIP2"
            )

        encoded_image = self.model.encode_image(image)
        output = self.model.answer_question(
            question=prompt,
            image_embeds=encoded_image,
            tokenizer=self.tokenizer,
            **generate_kwargs,
        )

        return output

    def infer_batch(self, images: list[str], prompts: list[str], **generate_kwargs):
        images = [
            Image.open(requests.get(image, stream=True).raw).convert("RGB")
            for image in images
        ]
        prompts = [prompt for prompt in prompts]

        outputs = self.model.batch_answer(
            images, prompts, self.tokenizer, **generate_kwargs
        )

        return outputs
