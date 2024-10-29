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
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class Llama32VisionInstruct(BaseModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        # Use device_map="auto" to automatically distribute model across available devices
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.dtype, device_map="auto", **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    @track_inference
    def infer(self, image: str, prompt: str, **generate_kwargs) -> str:
        image = super().parse_images(image)

        # Create messages format and apply chat template
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }
        ]
        input_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        # Process inputs without adding special tokens
        inputs = self.processor(
            image, input_text, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)

        decoded = self.processor.decode(output[0], skip_special_tokens=True)
        # Remove the prompt and assistant marker
        if "assistant" in decoded:
            decoded = decoded.split("assistant")[-1]
        return decoded.strip()

    def infer_batch(self, images: list[str], prompts: list[str], **generate_kwargs):
        images = super().parse_images(images)

        # Create batch messages and apply chat template
        messages_list = [
            [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": prompt}],
                }
            ]
            for prompt in prompts
        ]
        input_texts = [
            self.processor.apply_chat_template(msgs, add_generation_prompt=True)
            for msgs in messages_list
        ]

        inputs = self.processor(
            images, input_texts, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
        # Remove the prompt and assistant marker for each response
        return [d.split("assistant")[-1].strip() for d in decoded]


@register_model(
    "meta-llama/Llama-3.2-11B-Vision",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "meta-llama/Llama-3.2-90B-Vision",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class Llama32Vision(Llama32VisionInstruct):
    @track_inference
    def infer(self, image: str, prompt: str, **generate_kwargs) -> str:
        image = super().parse_images(image)

        # Format prompt for base vision model
        input_text = f"<|image|><|begin_of_text|>{prompt}"

        # Process inputs without adding special tokens
        inputs = self.processor(image, input_text, return_tensors="pt").to(
            self.model.device
        )

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generate_kwargs)

        return self.processor.decode(output[0], skip_special_tokens=True)

    def infer_batch(self, images: list[str], prompts: list[str], **generate_kwargs):
        images = super().parse_images(images)

        # Format prompts for base vision model
        input_texts = [f"<|image|><|begin_of_text|>{prompt}" for prompt in prompts]

        inputs = self.processor(images, input_texts, return_tensors="pt").to(
            self.model.device
        )

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        return self.processor.batch_decode(outputs, skip_special_tokens=True)
