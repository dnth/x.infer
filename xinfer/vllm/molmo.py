import requests
from PIL import Image
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

    def preprocess(
        self,
        images: str | list[str],
    ) -> list[Image.Image]:
        """
        Preprocess one or more images from file paths or URLs.

        Loads and converts images to RGB format from either local file paths or URLs.
        Can handle both single image input or multiple images as a list.

        Args:
            images (Union[str, List[str]]): Either a single image path/URL as a string,
                or a list of image paths/URLs. Accepts both local file paths and HTTP(S) URLs.

        Returns:
            List[PIL.Image.Image]: List of processed PIL Image objects in RGB format.
        """

        if not isinstance(images, list):
            images = [images]

        processed_images = []
        for image_path in images:
            if not isinstance(image_path, str):
                raise ValueError("Input must be a string (local path or URL)")

            if image_path.startswith(("http://", "https://")):
                image = Image.open(requests.get(image_path, stream=True).raw).convert(
                    "RGB"
                )
            else:
                # Assume it's a local path
                try:
                    image = Image.open(image_path).convert("RGB")
                except FileNotFoundError:
                    raise ValueError(f"Local file not found: {image_path}")

            processed_images.append(image)

        return processed_images

    def load_model(self, **kwargs):
        self.model = LLM(
            # model=self.model_id,
            model="/home/dnth/Desktop/cv-docker-images/image_captioning/molmo/molmo_7b_d_0924",
            trust_remote_code=True,
            dtype=self.dtype,
            **kwargs,
        )

    def infer_batch(self, images: list[str], prompts: list[str], **sampling_kwargs):
        images = self.preprocess(images)

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
            image = self.preprocess(image)

            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }

            sampling_params = SamplingParams(**sampling_kwargs)
            outputs = self.model.generate(inputs, sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
        self.update_inference_count(1)
        return generated_text
