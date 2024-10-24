from vllm import LLM, SamplingParams

from ..model_registry import ModelInputOutput, register_model
from ..models import BaseModel, track_inference


@register_model("allenai/Molmo-72B-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT)
@register_model("allenai/Molmo-7B-O-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT)
@register_model("allenai/Molmo-7B-D-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT)
class Molmo(BaseModel):
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
            model=self.model_id,
            trust_remote_code=True,
            dtype=self.dtype,
            **kwargs,
        )

    @track_inference
    def infer_batch(self, images: list[str], prompts: list[str], **sampling_kwargs):
        images = self.parse_images(images)

        sampling_params = SamplingParams(**sampling_kwargs)
        batch_inputs = [
            {
                "prompt": f"USER: <image>\n{prompt}\nASSISTANT:",
                "multi_modal_data": {"image": image},
            }
            for image, prompt in zip(images, prompts)
        ]

        results = self.model.generate(batch_inputs, sampling_params)

        return [output.outputs[0].text.strip() for output in results]

    @track_inference
    def infer(self, image: str, prompt: str, **sampling_kwargs):
        image = self.parse_images(image)

        inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
        }

        sampling_params = SamplingParams(**sampling_kwargs)
        outputs = self.model.generate(inputs, sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()
        return generated_text
