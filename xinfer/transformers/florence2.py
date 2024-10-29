import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from ..model_registry import ModelInputOutput, register_model
from ..models import BaseModel, track_inference


@register_model(
    "microsoft/Florence-2-large", "transformers", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
class Florence2(BaseModel):
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
    def infer(self, image: str, prompt: str = None, **generate_kwargs) -> str:
        image = self.parse_images(image)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.dtype
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                **generate_kwargs,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(image[0].width, image[0].height),
        )

        return parsed_answer

    @track_inference
    def infer_batch(
        self, images: list[str], prompts: list[str] = None, **generate_kwargs
    ) -> list[str]:
        images = self.parse_images(images)
        inputs = self.processor(text=prompts, images=images, return_tensors="pt").to(
            self.device, self.dtype
        )

        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                **generate_kwargs,
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )

        parsed_answers = [
            self.processor.post_process_generation(
                text, task=prompt, image_size=(img.width, img.height)
            )
            for text, prompt, img in zip(generated_text, prompts, images)
        ]

        return parsed_answers
