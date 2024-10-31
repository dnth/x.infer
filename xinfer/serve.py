from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from ray import serve

from .core import create_model

app = FastAPI()


class InferRequest(BaseModel):
    url: str
    prompt: str
    infer_kwargs: dict = {}


class InferBatchRequest(BaseModel):
    urls: list[str]
    prompts: list[str]
    infer_kwargs: dict = {}


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
                request.url, prompt=request.prompt, **request.infer_kwargs
            )
            return {"response": result}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

    @app.post("/infer_batch")
    async def infer_batch(self, request: InferBatchRequest) -> list[dict]:
        try:
            result = self.model.infer_batch(
                request.urls, request.prompts, **request.infer_kwargs
            )
            return [{"response": r} for r in result]
        except Exception as e:
            return [{"error": f"An error occurred: {str(e)}"}]


def serve_model(
    model_id: str,
    *,  # Force keyword arguments after model_id
    deployment_kwargs: dict = None,
    **model_kwargs,
):
    deployment_kwargs = deployment_kwargs or {}

    # If device is cuda, automatically add GPU requirement
    if model_kwargs.get("device") == "cuda":
        ray_actor_options = deployment_kwargs.get("ray_actor_options", {})
        ray_actor_options["num_gpus"] = ray_actor_options.get("num_gpus", 1)
        deployment_kwargs["ray_actor_options"] = ray_actor_options

    deployment = serve.deployment(**deployment_kwargs)(XInferModel)
    app = deployment.bind(model_id, **model_kwargs)

    try:
        serve.run(app, blocking=True)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Receiving shutdown signal. Cleaning up...")
    finally:
        serve.shutdown()
