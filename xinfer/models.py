import time
from abc import ABC, abstractmethod
from contextlib import contextmanager

import torch
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table


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
        }
        if dtype not in dtype_map:
            raise ValueError("dtype must be one of 'float32', 'float16', or 'bfloat16'")
        self.dtype = dtype_map[dtype]

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def infer(self, image: str, prompt: str):
        pass

    @abstractmethod
    def infer_batch(self, images: list[str], prompts: list[str]):
        pass

    def launch_gradio(self, **gradio_launch_kwargs):
        # Importing here to avoid circular import
        from .viz import launch_gradio

        launch_gradio(self, **gradio_launch_kwargs)

    @contextmanager
    def track_inference_time(self):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self.total_inference_time += (end_time - start_time) * 1000

    def update_inference_count(self, count: int):
        self.num_inferences += count
        self.average_latency = (
            self.total_inference_time / self.num_inferences
            if self.num_inferences
            else 0.0
        )

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
