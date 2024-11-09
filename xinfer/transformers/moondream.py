import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..model_registry import register_model
from ..models import BaseXInferModel, track_inference
from ..types import ModelInputOutput, Result


@register_model(
    "vikhyatk/moondream2", "transformers", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
class Moondream(BaseXInferModel):
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

        self.model.eval()
        self.model = torch.compile(self.model, mode="max-autotune")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    @track_inference
    def infer(self, image: str, text: str = None, **generate_kwargs) -> Result:
        image = self.parse_images(image)
        encoded_image = self.model.encode_image(image)
        output = self.model.answer_question(
            question=text,
            image_embeds=encoded_image,
            tokenizer=self.tokenizer,
            **generate_kwargs,
        )

        return Result(text=output)

    @track_inference
    def infer_batch(
        self, images: list[str], text: list[str], **generate_kwargs
    ) -> list[Result]:
        images = self.parse_images(images)
        text = [prompt for prompt in text]

        outputs = self.model.batch_answer(
            images, text, self.tokenizer, **generate_kwargs
        )

        return [Result(text=output) for output in outputs]
