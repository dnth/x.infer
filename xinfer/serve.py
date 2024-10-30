import signal
import sys
from typing import Dict

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from ray import serve

from .core import create_model

app = FastAPI()


class InferRequest(BaseModel):
    url: str
    prompt: str
    kwargs: Dict = {}


class InferBatchRequest(BaseModel):
    urls: list[str]
    prompts: list[str]
    kwargs: Dict = {}


@serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class XInferModel:
    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        self.model = create_model(model_id, **kwargs)

    @app.post("/infer")
    async def infer(self, request: InferRequest) -> Dict:
        try:
            result = self.model.infer(
                request.url, prompt=request.prompt, **request.kwargs
            )
            return {"response": result}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

    @app.post("/infer_batch")
    async def infer_batch(self, request: InferBatchRequest) -> list[Dict]:
        try:
            result = self.model.infer_batch(
                request.urls, request.prompts, **request.kwargs
            )
            return [{"response": r} for r in result]
        except Exception as e:
            return [{"error": f"An error occurred: {str(e)}"}]


def signal_handler(signum, frame):
    logger.info("\nReceiving shutdown signal. Cleaning up...")
    serve.shutdown()
    sys.exit(0)


def serve_model(model_id, **kwargs):
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    model_instance = XInferModel.bind(model_id, **kwargs)

    try:
        serve.run(model_instance, blocking=True)
    except KeyboardInterrupt:
        logger.info("\nReceiving keyboard interrupt. Cleaning up...")
        serve.shutdown()
        sys.exit(0)
