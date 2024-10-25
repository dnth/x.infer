import functools
import time
from abc import ABC, abstractmethod

import requests
import torch
from loguru import logger
from PIL import Image
from rich import box
from rich.console import Console
from rich.table import Table


def track_inference(func):
    """
    Decorator to track the inference time of a model. Put this on the inference methods of a model.

    Example:
        @track_inference
        def infer(self, image: str, prompt: str):
            ...
            return result

        @track_inference
        def infer_batch(self, images: list[str], prompts: list[str]):
            ...
            return result
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()

        inference_time = (end_time - start_time) * 1000
        self.total_inference_time += inference_time

        if func.__name__ == "infer_batch":
            # For batch inference, increment by the number of images
            num_inferences = len(args[0]) if args else len(kwargs.get("images", []))
        else:
            # For single inference, increment by 1
            num_inferences = 1

        self.num_inferences += num_inferences
        self.average_latency = self.total_inference_time / self.num_inferences

        return result

    return wrapper


class BaseModel(ABC):
    def __init__(self, model_id: str, device: str, dtype: str):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.num_inferences = 0
        self.total_inference_time = 0.0
        self.average_latency = 0.0

        logger.info(f"Model: {model_id}")
        logger.info(f"Device: {device}")
        logger.info(f"Dtype: {dtype}")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "auto": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError("dtype must be one of 'float32', 'float16', or 'bfloat16'")
        self.dtype = dtype_map[dtype]

    @abstractmethod
    def load_model(self):
        """
        Load the model and any necessary components.

        This method should be implemented by subclasses to initialize
        their specific model architecture and weights.

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        logger.info("Loading model...")
        raise NotImplementedError("Subclass must implement load_model()")

    @abstractmethod
    def infer(self, image: str, prompt: str):
        """
        Run inference on a single image.

        Args:
            image (str): Path or URL to the input image
            prompt (str): Text prompt for the inference

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        logger.info("Running single inference...")
        raise NotImplementedError("Subclass must implement infer()")

    @abstractmethod
    def infer_batch(self, images: list[str], prompts: list[str]):
        """
        Run inference on a batch of images.

        Args:
            images (list[str]): List of image paths or URLs
            prompts (list[str]): List of text prompts for each image

        Raises:
            NotImplementedError: If the subclass doesn't implement this method
        """
        logger.info("Running batch inference...")
        raise NotImplementedError("Subclass must implement infer_batch()")

    def launch_gradio(self, **gradio_launch_kwargs):
        # Importing here to avoid circular import
        from .viz import launch_gradio

        launch_gradio(self, **gradio_launch_kwargs)

    def print_stats(self):
        console = Console()
        table = Table(title="Model Info", box=box.ROUNDED)
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Model ID", str(self.model_id))
        table.add_row("Device", str(self.device))
        table.add_row("Dtype", str(self.dtype))
        table.add_row("Number of Inferences", f"{self.num_inferences}")
        table.add_row("Total Inference Time (ms)", f"{self.total_inference_time:.4f}")
        table.add_row("Average Latency (ms)", f"{self.average_latency:.4f}")

        console.print(table)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_id='{self.model_id}', "
            f"device='{self.device}', "
            f"dtype='{self.dtype}', "
        )

    def parse_images(
        self,
        images: str | list[str],
    ) -> list[Image.Image]:
        """
        Preprocess one or more images from file paths or URLs.

        Loads and converts images to RGB format from either local file paths or URLs.
        Can handle both single image input or multiple images as a list.

        Args:
            images (Union[str, List[str]]): Either a single image path/URL as a string,
                or a list of image paths/URLs. Accepts both local file paths and HTTP(S) URLs.

        Returns:
            List[PIL.Image.Image]: List of processed PIL Image objects in RGB format.
        """

        if not isinstance(images, list):
            images = [images]

        parsed_images = []
        for image_path in images:
            if not isinstance(image_path, str):
                raise ValueError("Input must be a string (local path or URL)")

            if image_path.startswith(("http://", "https://")):
                image = Image.open(requests.get(image_path, stream=True).raw).convert(
                    "RGB"
                )
            else:
                # Assume it's a local path
                try:
                    image = Image.open(image_path).convert("RGB")
                except FileNotFoundError:
                    raise ValueError(f"Local file not found: {image_path}")

            parsed_images.append(image)

        return parsed_images
