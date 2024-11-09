import ollama

from ..model_registry import register_model
from ..models import BaseXInferModel
from ..types import ModelInputOutput, Result


@register_model("ollama/llava-phi3", "ollama", ModelInputOutput.IMAGE_TEXT_TO_TEXT)
class LLaVAPhi3(BaseXInferModel):
    def __init__(
        self,
        model_id: str = "llava-phi3",
        device: str = "auto",
        dtype: str = "auto",
    ):
        super().__init__(model_id, device, dtype)
        self.load_model()

    def load_model(self):
        ollama.pull(self.model_id.replace("ollama/", ""))

    def infer_batch(self, image: str, prompt: str):
        raise NotImplementedError("Ollama models do not support batch inference")

    def infer(self, image: str, text: str) -> Result:
        res = ollama.chat(
            model="llava-phi3",
            messages=[
                {
                    "role": "user",
                    "content": text,
                    "images": [image],
                }
            ],
        )

        result = res["message"]["content"].strip()

        return Result(text=result)
