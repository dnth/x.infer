import gradio as gr

from .model_registry import ModelInputOutput, model_registry
from .models import BaseModel


def launch_gradio(model: BaseModel):
    model_info = model_registry.get_model_info(model.model_id)

    def infer(image, prompt=None):
        if prompt is not None:
            result = model.infer(image, prompt)
        else:
            result = model.infer(image)

        return result

    if model_info.input_output == ModelInputOutput.IMAGE_TEXT_TO_TEXT:
        iface = gr.Interface(
            fn=infer,
            inputs=[gr.Image(type="filepath"), gr.Textbox(label="Prompt")],
            outputs=gr.Textbox(label="Generated Text"),
            title=f"Inference with {model.model_id}",
            description="Upload an image and provide a prompt to generate a description.",
        )

    elif model_info.input_output == ModelInputOutput.IMAGE_TO_OBJECTS:
        iface = gr.Interface(
            fn=infer,
            inputs=gr.Image(type="filepath"),
            outputs=gr.JSON(label="Detection Results"),
            title=f"Object Detection with {model.model_id}",
            description="Upload an image to detect objects.",
        )

    elif model_info.input_output == ModelInputOutput.IMAGE_TO_CLASS:
        iface = gr.Interface(
            fn=infer,
            inputs=gr.Image(type="filepath"),
            outputs=gr.JSON(label="Classification Result"),
            title=f"Image Classification with {model.model_id}",
            description="Upload an image to classify.",
        )

    iface.launch()
