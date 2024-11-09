from vllm import LLM, SamplingParams

from ..model_registry import register_model
from ..models import BaseXInferModel, track_inference
from ..types import ModelInputOutput, Result


@register_model(
    "vllm/allenai/Molmo-72B-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
@register_model(
    "vllm/allenai/Molmo-7B-O-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
@register_model(
    "vllm/allenai/Molmo-7B-D-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT
)
class Molmo(BaseXInferModel):
    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.model = LLM(
            model=self.model_id.replace("vllm/", ""),
            trust_remote_code=True,
            dtype=self.dtype,
            max_model_len=4096,
            **kwargs,
        )

    @track_inference
    def infer_batch(
        self, images: list[str], texts: list[str], **sampling_kwargs
    ) -> list[Result]:
        images = self.parse_images(images)

        sampling_params = SamplingParams(**sampling_kwargs)
        batch_inputs = [
            {
                "prompt": f"USER: <image>\n{prompt}\nASSISTANT:",
                "multi_modal_data": {"image": image},
            }
            for image, prompt in zip(images, texts)
        ]

        results = self.model.generate(batch_inputs, sampling_params)

        return [Result(text=output.outputs[0].text.strip()) for output in results]

    @track_inference
    def infer(self, image: str, text: str, **sampling_kwargs) -> Result:
        image = self.parse_images(image)

        inputs = {
            "prompt": text,
            "multi_modal_data": {"image": image},
        }

        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self.model.generate(inputs, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        return Result(text=generated_text)
