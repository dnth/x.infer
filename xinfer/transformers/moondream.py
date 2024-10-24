import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model_registry import ModelInputOutput, register_model
from ..models import BaseModel, track_inference


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

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, revision=self.revision
        ).to(self.device, self.dtype)

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @track_inference
    def infer(self, image: str, prompt: str = None, **generate_kwargs):
        image = self.parse_images(image)
        encoded_image = self.model.encode_image(image)
        output = self.model.answer_question(
            question=prompt,
            image_embeds=encoded_image,
            tokenizer=self.tokenizer,
            **generate_kwargs,
        )

        return output

    @track_inference
    def infer_batch(self, images: list[str], prompts: list[str], **generate_kwargs):
        images = self.parse_images(images)
        prompts = [prompt for prompt in prompts]

        outputs = self.model.batch_answer(
            images, prompts, self.tokenizer, **generate_kwargs
        )

        return outputs
