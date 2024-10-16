import time
from abc import ABC, abstractmethod
from contextlib import contextmanager

import gradio as gr
from rich import box
from rich.console import Console
from rich.table import Table


class BaseModel(ABC):
    def __init__(self, model_id: str, device: str, dtype: str):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        self.stats = ModelStats(model_id, device, dtype)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def infer(self, image: str, prompt: str):
        pass

    @abstractmethod
    def infer_batch(self, images: list[str], prompts: list[str]):
        pass

    def launch_gradio(self):
        def inference(image):
            if hasattr(self, "infer"):
                result = self.infer(image)
                if isinstance(result, list):
                    return str(result)  # Convert list to string for display
                return result
            else:
                return "Infer method not implemented for this model."

        inputs = [gr.Image(type="filepath")]
        outputs = gr.Textbox()

        # Add prompt input if the model's infer method accepts it
        if "prompt" in self.infer.__code__.co_varnames:
            inputs.append(gr.Textbox(label="Prompt"))
            inference = lambda image, prompt: self.infer(image, prompt)

        iface = gr.Interface(
            fn=inference,
            inputs=inputs,
            outputs=outputs,
            title=f"{self.model_id} Inference",
            description="Upload an image to get the model's output.",
        )
        iface.launch()


class ModelStats:
    def __init__(
        self,
        model_id: str,
        device: str,
        dtype: str,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.num_inferences = 0
        self.total_inference_time = 0.0
        self.average_latency = 0.0

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
        table = Table(title="Model Stats", box=box.ROUNDED)
        table.add_column("Attribute", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Model ID", str(self.model_id))
        table.add_row("Device", str(self.device))
        table.add_row("Dtype", str(self.dtype))
        table.add_row("Number of Inferences", f"{self.num_inferences}")
        table.add_row("Total Inference Time (ms)", f"{self.total_inference_time:.4f}")
        table.add_row("Average Latency (ms)", f"{self.average_latency:.4f}")

        console.print(table)
