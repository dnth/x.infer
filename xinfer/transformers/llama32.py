import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

from ..model_registry import ModelInputOutput, register_model
from ..models import BaseModel, track_inference


@register_model(
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "meta-llama/Llama-3.2-11B-Vision",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "meta-llama/Llama-3.2-90B-Vision",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class Llama32(BaseModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, **kwargs
        ).to(self.device, self.dtype)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    @track_inference
    def infer(self, image: str, prompt: str, **generate_kwargs) -> str:
        image = super().parse_images(image)
        model_prompt = f"<|image|><|begin_of_text|>{prompt}"
        inputs = self.processor(image, model_prompt, return_tensors="pt").to(
            self.model.device
        )

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)

        decoded = self.processor.decode(output[0], skip_special_tokens=True)

        return decoded[len(prompt) :].strip()

    def infer_batch(self, images: list[str], prompts: list[str]):
        raise NotImplementedError("Batch inference not supported for this model")
