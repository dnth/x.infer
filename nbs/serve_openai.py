import time
from io import BytesIO
from typing import Dict, List

import numpy as np
import requests
import timm
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from ray import serve

app = FastAPI()


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model("resnet50", pretrained=True).to(self.device)
        self.model.eval()

        # Load ImageNet class names
        url = (
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        self.classes = requests.get(url).text.splitlines()

    async def classify_image(self, image_url: str) -> str:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))
        img_tensor = (
            torch.from_numpy(np.array(img))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        img_tensor /= 255.0

        with torch.no_grad():
            output = self.model(img_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        result = [
            f"{self.classes[top5_catid[i]]}: {top5_prob[i].item():.2%}"
            for i in range(5)
        ]
        return "\n".join(result)

    @app.post("/v1/chat/completions")
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        if request.model != "image-classifier":
            raise HTTPException(status_code=400, detail="Unsupported model")

        last_message = request.messages[-1]
        if last_message["role"] != "user":
            raise HTTPException(
                status_code=400, detail="Last message must be from user"
            )

        image_url = last_message["content"]
        try:
            classification_result = await self.classify_image(image_url)

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error classifying image: {str(e)}"
            )

        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=int(time.time()),
            model="image-classifier",
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"Here are the top 5 classifications for the image {classification_result}",
                    },
                    "finish_reason": "stop",
                }
            ],
        )
        return response


deployment = ImageClassifier.bind()

if __name__ == "__main__":
    import subprocess

    # Run the serve deployment command
    subprocess.run(["serve", "run", "serve:deployment"], check=True)
