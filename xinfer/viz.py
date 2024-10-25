import json

import gradio as gr

from .core import create_model
from .model_registry import ModelInputOutput, model_registry
from .models import BaseModel


def launch_gradio(model: BaseModel, **gradio_launch_kwargs):
    model_info = model_registry.get_model_info(model.model_id)

    def infer(image, prompt=None):
        try:
            if prompt is not None:
                result = model.infer(image, prompt)
            else:
                result = model.infer(image)

            # Convert result to string if it's not already
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error during inference: {str(e)}"

    if model_info.input_output == ModelInputOutput.IMAGE_TEXT_TO_TEXT:
        iface = gr.Interface(
            fn=infer,
            inputs=[gr.Image(type="filepath"), gr.Textbox(label="Prompt")],
            outputs=gr.Textbox(label="Result", lines=10),
            title=f"Inference with {model.model_id}",
            description="Upload an image and provide a prompt to generate a description.",
        )

    elif model_info.input_output == ModelInputOutput.IMAGE_TO_BOXES:
        iface = gr.Interface(
            fn=infer,
            inputs=gr.Image(type="filepath"),
            outputs=gr.Textbox(label="Result", lines=10),
            title=f"Object Detection with {model.model_id}",
            description="Upload an image to detect objects.",
        )

    elif model_info.input_output == ModelInputOutput.IMAGE_TO_CATEGORIES:
        iface = gr.Interface(
            fn=infer,
            inputs=gr.Image(type="filepath"),
            outputs=gr.Textbox(label="Result", lines=10),
            title=f"Image Classification with {model.model_id}",
            description="Upload an image to classify.",
        )

    # The default height of Gradio is too small for view in jupyter notebooks
    if "height" not in gradio_launch_kwargs:
        gradio_launch_kwargs["height"] = 1000

    iface.launch(**gradio_launch_kwargs)


def launch_gradio_demo():
    """
    Launch an interactive demo with a dropdown to select a model from all supported models,
    and a button to run inference.
    """
    available_models = [model.id for model in model_registry.list_models()]

    # Add example image URLs
    example_images = [
        "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/000b9c365c9e307a.jpg",
        "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/00aa2580828a9009.jpg",
        "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/0a6ee446579d2885.jpg",
    ]

    def load_model_and_infer(model_id, image, prompt, device, dtype):
        model = create_model(model_id, device=device, dtype=dtype)
        model_info = model_registry.get_model_info(model_id)

        try:
            if model_info.input_output == ModelInputOutput.IMAGE_TEXT_TO_TEXT:
                result = model.infer(image, prompt)
            elif model_info.input_output in [
                ModelInputOutput.IMAGE_TO_BOXES,
                ModelInputOutput.IMAGE_TO_CATEGORIES,
            ]:
                result = model.infer(image)
            else:
                return "Unsupported model type"

            # Convert result to string if it's not already
            if isinstance(result, str):
                return result
            else:
                return json.dumps(result, indent=2)
        except Exception as e:
            return f"Error during inference: {str(e)}"

    with gr.Blocks() as demo:
        gr.Markdown("# x.infer Gradio Demo")

        with gr.Row():
            # Left column: Input controls
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=available_models, label="Select a model"
                )
                with gr.Row():
                    device_dropdown = gr.Dropdown(
                        choices=["cuda", "cpu"], label="Device", value="cuda"
                    )
                    dtype_dropdown = gr.Dropdown(
                        choices=["float32", "float16", "bfloat16"],
                        label="Dtype",
                        value="float16",
                    )
                prompt_input = gr.Textbox(
                    label="Prompt (for image-text to text models)", visible=False
                )
                run_button = gr.Button("Run Inference", variant="primary")

            # Right column: Image input
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Input Image", height=400)

        # Results section
        output = gr.Textbox(label="Result", lines=5)

        # Add examples
        gr.Examples(examples=example_images, inputs=image_input, label="Example Images")

        def update_prompt_visibility(model_id):
            model_info = model_registry.get_model_info(model_id)
            return gr.update(
                visible=model_info.input_output == ModelInputOutput.IMAGE_TEXT_TO_TEXT
            )

        model_dropdown.change(
            update_prompt_visibility, inputs=[model_dropdown], outputs=[prompt_input]
        )

        run_button.click(
            load_model_and_infer,
            inputs=[
                model_dropdown,
                image_input,
                prompt_input,
                device_dropdown,
                dtype_dropdown,
            ],
            outputs=[output],
        )

    demo.launch(height=1000)
