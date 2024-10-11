import requests
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from ..base_model import BaseModel


class BLIP2(BaseModel):
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", **kwargs):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model(**kwargs)

    def load_model(self, **kwargs):
        self.processor = Blip2Processor.from_pretrained(self.model_name, **kwargs)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        ).to(self.device, torch.bfloat16)

        self.model = torch.compile(self.model, mode="max-autotune")
        self.model.eval()

    def preprocess(self, image: str | Image.Image, prompt: str = None):
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                raise ValueError("Input string must be an image URL for BLIP2")
        else:
            raise ValueError(
                "Input must be either an image URL or a PIL Image for BLIP2"
            )

        return self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )

    def predict(self, preprocessed_input, **generate_kwargs):
        with torch.inference_mode(), torch.amp.autocast(
            device_type=self.device, dtype=torch.bfloat16
        ):
            return self.model.generate(**preprocessed_input, **generate_kwargs)

    def postprocess(self, prediction):
        output = self.processor.batch_decode(prediction, skip_special_tokens=True)[0]
        return output.replace("\n", "").strip()

    def inference(self, image, prompt, **generate_kwargs):
        preprocessed_input = self.preprocess(image, prompt)
        prediction = self.predict(preprocessed_input, **generate_kwargs)
        return self.postprocess(prediction)


class VLRMBlip2(BLIP2):
    def __init__(self, model_name: str = "sashakunitsyn/vlrm-blip2-opt-2.7b", **kwargs):
        super().__init__(model_name, **kwargs)
        self.load_vlrm_weights()

    def load_vlrm_weights(self):
        from huggingface_hub import hf_hub_download

        finetuned_weights_state_dict = torch.load(
            hf_hub_download(
                repo_id="sashakunitsyn/vlrm-blip2-opt-2.7b",
                filename="vlrm-blip2-opt-2.7b.pt",
            )
        )
        self.model.load_state_dict(finetuned_weights_state_dict, strict=False)
