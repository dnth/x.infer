from ..model_registry import register_model
from ..models import track_inference
from ..types import ModelInputOutput, Result
from .vision2seq import Vision2SeqModel


@register_model(
    "HuggingFaceTB/SmolVLM-Instruct",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class SmolVLM(Vision2SeqModel):
    def __init__(self, model_id: str, device: str = "cpu", dtype: str = "float32"):
        super().__init__(model_id, device, dtype)

    @track_inference
    def infer(self, image: str, text: str, **generate_kwargs) -> Result:
        if "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 300

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            }
        ]

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = inputs.to(self.device)

        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Extract only the assistant's response
        response = generated_texts[0].split("Assistant:", 1)[-1].strip()
        return Result(text=response)

    def infer_batch(self, *args, **kwargs):
        raise NotImplementedError("SmolVLM does not support batch inference.")
