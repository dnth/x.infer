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
        model_name: str = "vikhyatk/moondream2",
        revision: str = "2024-08-26",
        **kwargs,
    ):
        self.model_name = model_name
        self.revision = revision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, trust_remote_code=True, revision=self.revision
        )

        if self.device == "cuda":
            self.model = self.model.to(self.device, torch.bfloat16)

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def inference(self, image: str, prompt: str = None, **generate_kwargs):
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
