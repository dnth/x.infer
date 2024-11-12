import inspect
import time
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
from ray import serve

from .core import create_model

app = FastAPI()


class InferRequest(BaseModel):
    image: str
    infer_kwargs: dict = {}


class InferBatchRequest(BaseModel):
    images: list[str]
    infer_batch_kwargs: dict = {}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{time.time_ns()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[dict]
    usage: dict = Field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )


@serve.ingress(app)
class XInferModel:
    def __init__(
        self,
        model_id,
        **kwargs,
    ):
        try:
            self.model = create_model(model_id, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_id}: {str(e)}")

    @app.post("/infer")
    async def infer(self, request: InferRequest) -> dict:
        try:
            result = self.model.infer(request.image, **request.infer_kwargs)
            return {"response": result}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

    @app.post("/infer_batch")
    async def infer_batch(self, request: InferBatchRequest) -> list[dict]:
        try:
            result = self.model.infer_batch(
                request.images, **request.infer_batch_kwargs
            )
            return [{"response": r} for r in result]
        except Exception as e:
            return [{"error": f"An error occurred: {str(e)}"}]

    @app.get("/health")
    async def health(self):
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "model_id": self.model.model_id,
            "device": self.model.device,
            "dtype": str(self.model.dtype),
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        try:
            # Extract the image URL and prompt from the OpenAI format
            message = request.messages[-1]["content"]
            infer_kwargs = {}

            if isinstance(message, list):
                # Extract image URL
                image_url = next(
                    m["image_url"] for m in message if m["type"] == "image_url"
                )

                # Extract infer_kwargs if present
                infer_kwargs_item = next(
                    (m["infer_kwargs"] for m in message if m["type"] == "infer_kwargs"),
                    None,
                )
                if infer_kwargs_item:
                    infer_kwargs.update(infer_kwargs_item)

            else:
                raise ValueError("Image URL is required in the message")

            # Call the model's infer method with kwargs
            result = self.model.infer(image=image_url, **infer_kwargs)

            response = ChatCompletionResponse(
                model=self.model.model_id,
                choices=[
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result},
                        "finish_reason": "stop",
                    }
                ],
            )
            return response
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


def serve_model(
    model_id: str,
    *,  # Force keyword arguments after model_id
    deployment_kwargs: dict = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    blocking: bool = True,
    open_api_docs: bool = True,
    **model_kwargs,
):
    deployment_kwargs = deployment_kwargs or {}

    # If device is cuda, automatically add GPU requirement
    if model_kwargs.get("device") == "cuda":
        ray_actor_options = deployment_kwargs.get("ray_actor_options", {})
        ray_actor_options["num_gpus"] = ray_actor_options.get("num_gpus", 1)
        deployment_kwargs["ray_actor_options"] = ray_actor_options

    serve.start(http_options={"host": host, "port": port})

    deployment = serve.deployment(**deployment_kwargs)(XInferModel)
    app = deployment.bind(model_id, **model_kwargs)

    try:
        handle = serve.run(app)
        logger.info(f"Open FastAPI docs at http://{host}:{port}/docs")
        if open_api_docs:
            import webbrowser

            webbrowser.open(f"http://{host}:{port}/docs")

        if blocking:
            try:
                while True:
                    time.sleep(1)
            except (KeyboardInterrupt, SystemExit):
                logger.info("Receiving shutdown signal. Cleaning up...")
                serve.shutdown()
        else:
            logger.info(
                "Running server in non-blocking mode, remember to call serve.shutdown() to stop the server"
            )
            return handle
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        serve.shutdown()
        raise
