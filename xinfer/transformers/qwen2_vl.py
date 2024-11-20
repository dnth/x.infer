import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from ..model_registry import register_model
from ..models import BaseXInferModel, track_inference
from ..types import ModelInputOutput, Result


@register_model(
    "Qwen/Qwen2-VL-2B-Instruct",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-2B-Instruct-AWQ",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-7B-Instruct",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-7B-Instruct-AWQ",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-72B-Instruct",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-72B-Instruct-AWQ",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
@register_model(
    "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
    "transformers",
    ModelInputOutput.IMAGE_TEXT_TO_TEXT,
)
class Qwen2VL(BaseXInferModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.dtype, device_map="auto", **kwargs
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    @track_inference
    def infer(self, image: str, text: str, **generate_kwargs) -> Result:
        image = super().parse_images(image)

        if "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 128

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=self.dtype
        ):
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return Result(text=output_text[0])

    @track_inference
    def infer_batch(
        self, images: list[str], text: str, **generate_kwargs
    ) -> list[Result]:
        images = super().parse_images(images)

        if "max_new_tokens" not in generate_kwargs:
            generate_kwargs["max_new_tokens"] = 128

        # Create batch conversations
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text},
                    ],
                }
            ]
            for _ in range(len(images))
        ]

        # Apply chat template to all conversations
        text_prompts = [
            self.processor.apply_chat_template(conv, add_generation_prompt=True)
            for conv in conversations
        ]

        # Process batch inputs
        inputs = self.processor(
            text=text_prompts, images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)

        # Batch inference
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=self.dtype
        ):
            output_ids = self.model.generate(**inputs, **generate_kwargs)

        # Trim generated ids
        generated_ids = [
            output_ids[i][len(inputs.input_ids[i]) :] for i in range(len(output_ids))
        ]

        # Decode outputs
        output_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return [Result(text=text) for text in output_texts]
