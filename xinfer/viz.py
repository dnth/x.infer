import json

import gradio as gr
from PIL import Image, ImageDraw

from .core import create_model
from .model_registry import model_registry
from .models import BaseXInferModel
from .types import ModelInputOutput


def launch_gradio(model: BaseXInferModel, **gradio_launch_kwargs):
    model_info = model_registry.get_model_info(model.model_id)

    def infer(image, prompt=None):
        try:
            if prompt is not None:
                result = model.infer(image, prompt)
            else:
                result = model.infer(image)

            return str(result)
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

    example_images = [
        "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/000b9c365c9e307a.jpg",
        "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/00aa2580828a9009.jpg",
        "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/0a6ee446579d2885.jpg",
    ]

    model_cache = {
        "current_model": None,
        "model_id": None,
        "device": None,
        "dtype": None,
    }

    def visualize_predictions(image_path, result_dict):
        """Draw bounding boxes and masks on the image."""
        # Open image and convert to RGBA to support transparency
        image = Image.open(image_path).convert("RGBA")

        # Create a transparent overlay for the masks
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw boxes if present
        if "boxes" in result_dict:
            for box in result_dict["boxes"]:
                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                label = box["label"]
                score = box["score"]

                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # Draw label with score
                label_text = f"{label}: {score:.2f}"
                draw.text((x1, y1 - 10), label_text, fill="red")

        # Draw masks if present
        if "masks" in result_dict:
            for mask in result_dict["masks"]:
                # Convert list of [x,y] coordinates to flat list for polygon drawing
                xy_flat = [coord for point in mask["xy"] for coord in point]

                draw.polygon(xy_flat, outline="blue", fill=(0, 0, 255, 100))

        # Composite the original image with the overlay
        result_image = Image.alpha_composite(image, overlay)

        # Convert back to RGB for display
        return result_image.convert("RGB")

    def load_model_and_infer(model_id, image, text, device, dtype):
        # Check if we need to load a new model
        if (
            model_cache["model_id"] != model_id
            or model_cache["device"] != device
            or model_cache["dtype"] != dtype
        ):
            model = create_model(model_id, device=device, dtype=dtype)
            # Update cache
            model_cache.update(
                {
                    "current_model": model,
                    "model_id": model_id,
                    "device": device,
                    "dtype": dtype,
                }
            )
        else:
            model = model_cache["current_model"]

        model_info = model_registry.get_model_info(model_id)

        try:
            if requires_text_prompt(model_info):
                result = model.infer(image, text)
            else:
                result = model.infer(image)

            # Check if result contains boxes or masks
            try:
                result_dict = json.loads(str(result))
                if "boxes" in result_dict or "masks" in result_dict:
                    return visualize_predictions(image, result_dict), str(result)
            except json.JSONDecodeError:
                pass

            return None, str(result)
        except Exception as e:
            return None, f"Error during inference: {str(e)}"

    def requires_text_prompt(model_info):
        return model_info.input_output == ModelInputOutput.IMAGE_TEXT_TO_TEXT

    with gr.Blocks() as demo:
        gr.Markdown("# x.infer Gradio Demo")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Input Image", height=400)

                # Add examples
                gr.Examples(
                    examples=example_images, inputs=image_input, label="Example Images"
                )

            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    label="Select a model",
                    value="vikhyatk/moondream2",
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
                prompt_input = gr.Textbox(label="Text Prompt", visible=True)
                run_button = gr.Button("Run Inference", variant="primary")

        # Results section
        with gr.Row():
            output_text = gr.Textbox(label="Result", lines=5)

        output_image = gr.Image(label="Visualization")

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
            outputs=[output_image, output_text],
        )

    demo.launch(height=1000)
