import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ..model_registry import register_model
from ..models import BaseXInferModel, track_inference
from ..types import ModelInputOutput, Result


@register_model(
    "microsoft/Florence-2-large", "transformers", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
@register_model(
    "microsoft/Florence-2-base",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "microsoft/Florence-2-large-ft",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "microsoft/Florence-2-base-ft",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class Florence2(BaseXInferModel):
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        super().__init__(model_id, device, dtype)
        self.load_model()

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True
        ).to(self.device, self.dtype)
        self.model.eval()
        self.model = torch.compile(self.model, mode="max-autotune")
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )

    @track_inference
    def infer(self, image: str, text: str, **generate_kwargs) -> Result:
        output = self.infer_batch([image], [text], **generate_kwargs)
        return output[0]

    @track_inference
    def infer_batch(
        self, images: list[str], texts: list[str], **generate_kwargs
    ) -> list[Result]:
        images = self.parse_images(images)
        inputs = self.processor(text=texts, images=images, return_tensors="pt").to(
            self.device, self.dtype
        )

        if "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 1024
        if "num_beams" not in generate_kwargs:
            generate_kwargs["num_beams"] = 3

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                **generate_kwargs,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )

        parsed_answers = [
            self.processor.post_process_generation(
                text, task=prompt, image_size=(img.width, img.height)
            ).get(prompt)
            for text, prompt, img in zip(generated_text, texts, images)
        ]

        return [Result(text=text) for text in parsed_answers]
