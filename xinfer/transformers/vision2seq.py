import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from ..models import BaseXInferModel, track_inference
from ..types import Result


class Vision2SeqModel(BaseXInferModel):
    def __init__(
        self, model_id: str, device: str = "cpu", dtype: str = "float32", **kwargs
    ):
        super().__init__(model_id, device, dtype)
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs).to(
            self.device, self.dtype
        )

        self.model.eval()
        self.model = torch.compile(self.model, mode="max-autotune")

    def preprocess(
        self,
        images: str | list[str],
        texts: str | list[str],
    ):
        processed_images = self.parse_images(images)

        return self.processor(
            images=processed_images, text=texts, return_tensors="pt", padding=True
        ).to(self.device, self.dtype)

    def predict(self, preprocessed_input, **generate_kwargs):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=self.dtype
        ):
            return self.model.generate(**preprocessed_input, **generate_kwargs)

    def postprocess(self, predictions):
        outputs = self.processor.batch_decode(predictions, skip_special_tokens=True)
        return [output.replace("\n", "").strip() for output in outputs]

    @track_inference
    def infer(self, image: str, text: str, **generate_kwargs) -> Result:
        preprocessed_input = self.preprocess(image, text)
        prediction = self.predict(preprocessed_input, **generate_kwargs)
        result = self.postprocess(prediction)[0]

        return Result(text=result)

    @track_inference
    def infer_batch(
        self, images: list[str], texts: list[str], **generate_kwargs
    ) -> list[Result]:
        preprocessed_input = self.preprocess(images, texts)
        predictions = self.predict(preprocessed_input, **generate_kwargs)
        results = self.postprocess(predictions)

        return [Result(text=result) for result in results]
