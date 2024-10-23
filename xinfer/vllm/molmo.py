from vllm import LLM, SamplingParams

from ..model_registry import ModelInputOutput, register_model
from ..models import BaseModel


@register_model("allenai/Molmo-72B-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT)
@register_model("allenai/Molmo-7B-O-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT)
@register_model("allenai/Molmo-7B-D-0924", "vllm", ModelInputOutput.IMAGE_TEXT_TO_TEXT)
class Molmo(BaseModel):
    def __init__(
        self,
        model_id: str,
        device: str = "cuda",
        dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.model = LLM(
            # model=self.model_id,
            model="/home/dnth/Desktop/cv-docker-images/image_captioning/molmo/molmo_7b_d_0924",
            trust_remote_code=True,
            dtype=self.dtype,
            **kwargs,
        )

    def infer_batch(self, images: list[str], prompts: list[str], **sampling_kwargs):
        images = self.parse_images(images)

        sampling_params = SamplingParams(**sampling_kwargs)
        with self.track_inference_time():
            batch_inputs = [
                {
                    "prompt": f"USER: <image>\n{prompt}\nASSISTANT:",
                    "multi_modal_data": {"image": image},
                }
                for image, prompt in zip(images, prompts)
            ]

            results = self.model.generate(batch_inputs, sampling_params)

        self.update_inference_count(len(images))
        return [output.outputs[0].text.strip() for output in results]

    def infer(self, image: str, prompt: str, **sampling_kwargs):
        with self.track_inference_time():
            image = self.parse_images(image)

            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }

            sampling_params = SamplingParams(**sampling_kwargs)
            outputs = self.model.generate(inputs, sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
        self.update_inference_count(1)
        return generated_text
