from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel
from ray import serve

from .model_registry import model_registry

app = FastAPI()


class InferRequest(BaseModel):
    url: str
    prompt: str


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class XInferModel:
    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        self.model = model_registry.get_model(model_id, **kwargs)

    @app.post("/infer")
    async def infer(self, request: InferRequest) -> Dict:
        try:
            result = self.model.infer(request.url, prompt=request.prompt)
            return {"response": result}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}


def serve_model(model_id, **kwargs):
    model_instance = XInferModel.bind(model_id, **kwargs)

    serve.run(model_instance, blocking=True)
