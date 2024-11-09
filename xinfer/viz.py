import json

import gradio as gr
from PIL import Image, ImageDraw

from .core import create_model
from .model_registry import model_registry
from .models import BaseXInferModel
from .types import ModelInputOutput

example_images = [
    "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/000b9c365c9e307a.jpg",
    "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/00aa2580828a9009.jpg",
    "https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/demo/0a6ee446579d2885.jpg",
]


def visualize_predictions(image_path, result_dict):
    """Draw bounding boxes, masks, and poses on the image."""
    # Open image and convert to RGBA
    image = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw boxes if present
    if "boxes" in result_dict:
        for box in result_dict["boxes"]:
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            label = box["label"]
            score = box["score"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            label_text = f"{label}: {score:.2f}"
            draw.text((x1, y1 - 10), label_text, fill="red")

    # Draw masks if present
    if "masks" in result_dict:
        for mask in result_dict["masks"]:
            xy_flat = [coord for point in mask["xy"] for coord in point]
            draw.polygon(xy_flat, outline="blue", fill=(0, 0, 255, 32))

    # Draw poses if present
    if "poses" in result_dict:
        for pose in result_dict["poses"]:
            keypoints = pose["keypoints"][0]  # First set of keypoints
            scores = pose["scores"][0]

            # Draw each keypoint as a circle
            for i, (x, y) in enumerate(keypoints):
                score = scores[i]
                if (
                    score > 0.3 and x > 0 and y > 0
                ):  # Only draw if confidence > 0.3 and position is valid
                    # Draw a more visible point
                    radius = 5
                    draw.ellipse(
                        [x - radius, y - radius, x + radius, y + radius],
                        fill=(255, 0, 0, 255),  # Solid red
                        outline=(255, 255, 255, 255),  # White outline
                    )

            # Define connections for skeleton visualization
            # Example connections (you can modify these based on your needs)
            connections = [
                (5, 6),  # shoulders
                (5, 7),  # left arm
                (6, 8),  # right arm
                (7, 9),  # left forearm
                (8, 10),  # right forearm
                (5, 11),  # left torso
                (6, 12),  # right torso
                (11, 13),  # left thigh
                (12, 14),  # right thigh
                (13, 15),  # left leg
                (14, 16),  # right leg
            ]

            # Draw lines between connected keypoints
            for start_idx, end_idx in connections:
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                start_score = scores[start_idx]
                end_score = scores[end_idx]

                if (
                    start_score > 0.3
                    and end_score > 0.3
                    and all(start_point)
                    and all(end_point)
                ):  # Check if points are valid
                    draw.line(
                        [
                            start_point[0],
                            start_point[1],
                            end_point[0],
                            end_point[1],
                        ],
                        fill=(255, 165, 0, 255),  # Solid orange
                        width=3,
                    )

    # Composite the original image with the overlay
    result_image = Image.alpha_composite(image, overlay)
    return result_image.convert("RGB")


def launch_gradio(model: BaseXInferModel, **gradio_launch_kwargs):
    model_info = model_registry.get_model_info(model.model_id)

    def infer(image, prompt=None, device=None, dtype=None):
        try:
            # Create new model instance if device/dtype specified
            current_model = model
            if device is not None or dtype is not None:
                current_model = create_model(
                    model.model_id,
                    device=device or model.device,
                    dtype=dtype or model.dtype,
                )

            if prompt is not None:
                result = current_model.infer(image, prompt)
            else:
                result = current_model.infer(image)

            # Check if result contains visualization data
            try:
                result_dict = json.loads(str(result))
                if (
                    "boxes" in result_dict
                    or "masks" in result_dict
                    or "poses" in result_dict
                ):
                    return visualize_predictions(image, result_dict), str(result)
            except json.JSONDecodeError:
                pass

            return None, str(result)
        except Exception as e:
            return None, f"Error during inference: {str(e)}"

    with gr.Blocks() as iface:
        gr.Markdown(f"# Inference with {model.model_id}")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Input Image")

                gr.Examples(
                    examples=example_images, inputs=image_input, label="Example Images"
                )

            with gr.Column(scale=1):
                with gr.Row():
                    device_dropdown = gr.Dropdown(
                        choices=["cuda", "cpu"],
                        label="Device",
                        value=model.device,
                        interactive=False,
                    )
                    dtype_dropdown = gr.Dropdown(
                        choices=["float32", "float16", "bfloat16"],
                        label="Dtype",
                        value=str(model.dtype).split(".")[-1],
                        interactive=False,
                    )

                if model_info.input_output == ModelInputOutput.IMAGE_TEXT_TO_TEXT:
                    prompt_input = gr.Textbox(label="Prompt")
                else:
                    prompt_input = None

                run_button = gr.Button("Run Inference", variant="primary")

        with gr.Row():
            output_text = gr.Textbox(label="Result", lines=5)

        output_image = gr.Image(label="Visualization")

        # Set up the click event
        inputs = []
        if prompt_input is not None:
            inputs = [image_input, prompt_input, device_dropdown, dtype_dropdown]
        else:
            inputs = [image_input, device_dropdown, dtype_dropdown]

        # Fix the lambda function to handle arguments properly
        if prompt_input is not None:
            run_button.click(
                fn=lambda img, prompt, dev, dt: infer(img, prompt, dev, dt),
                inputs=inputs,
                outputs=[output_image, output_text],
            )
        else:
            run_button.click(
                fn=lambda img, dev, dt: infer(img, None, dev, dt),
                inputs=inputs,
                outputs=[output_image, output_text],
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

    model_cache = {
        "current_model": None,
        "model_id": None,
        "device": None,
        "dtype": None,
    }

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

            # Check if result contains boxes, masks, or poses
            try:
                result_dict = json.loads(str(result))
                if (
                    "boxes" in result_dict
                    or "masks" in result_dict
                    or "poses" in result_dict
                ):
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
