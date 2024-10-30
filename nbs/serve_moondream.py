from typing import Dict

import ray
from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

import xinfer

app = FastAPI()


class ImageRequest(BaseModel):
    url: str
    prompt: str


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class XInferModel:
    def __init__(self):
        self.model = xinfer.create_model(
            "vikhyatk/moondream2", device="cuda", dtype="float16"
        )

    @app.post("/infer")
    async def infer(self, request: ImageRequest) -> Dict:
        try:
            result = self.model.infer(request.url, prompt=request.prompt)
            return {"response": result}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}


image_model = XInferModel.bind()
serve.run(image_model, blocking=True)
