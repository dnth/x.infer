import signal
import sys

from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from ray import serve

from .core import create_model

app = FastAPI()


class InferRequest(BaseModel):
    url: str
    prompt: str
    kwargs: dict = {}


class InferBatchRequest(BaseModel):
    urls: list[str]
    prompts: list[str]
    kwargs: dict = {}


# @serve.deployment(num_replicas=1, ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class XInferModel:
    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        self.model = create_model(model_id, **kwargs)

    @app.post("/infer")
    async def infer(self, request: InferRequest) -> dict:
        try:
            result = self.model.infer(
                request.url, prompt=request.prompt, **request.kwargs
            )
            return {"response": result}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

    @app.post("/infer_batch")
    async def infer_batch(self, request: InferBatchRequest) -> list[dict]:
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


def serve_model(model_id: str, deployment_kwargs: dict = None, **model_kwargs):
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Set default deployment kwargs if none provided
    deployment_kwargs = deployment_kwargs or {
        "num_replicas": 1,
        "ray_actor_options": {"num_gpus": 1},
    }

    deployment = serve.deployment(**deployment_kwargs)(XInferModel)

    app = deployment.bind(model_id, **model_kwargs)

    try:
        serve.run(app, blocking=True)
    except KeyboardInterrupt:
        logger.info("\nReceiving keyboard interrupt. Cleaning up...")
        serve.shutdown()
        sys.exit(0)
