![PyPI version](https://img.shields.io/pypi/v/xinfer.svg?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=blue)
![Downloads](https://img.shields.io/pypi/dm/xinfer.svg?style=for-the-badge&logo=pypi&logoColor=white&label=Downloads&color=purple)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge&logo=apache&logoColor=white)


# xinfer
A unified interface (with focus on computer vision models) to run inference on machine learning libraries.


## Overview
xinfer is a modular Python framework that provides a unified interface for performing inference across a variety of machine learning models and libraries. Designed to simplify and standardize the inference process, xinfer allows developers to work seamlessly with models from Hugging Face Transformers, Ultralytics YOLO, and custom-built models using a consistent and easy-to-use API.

## Key Features
- Unified Interface: Interact with different machine learning models through a single, consistent API.
- Modular Design: Easily integrate and swap out models without altering the core framework.
- Flexible Architecture: Built using design patterns like Factory, Adapter, and Strategy for extensibility and maintainability.
- Ease of Use: Simplifies model loading, input preprocessing, inference execution, and output postprocessing.
- Extensibility: Add support for new models and libraries with minimal code changes.
- Robust Error Handling: Provides meaningful error messages and gracefully handles exceptions.


## Supported Libraries
- Hugging Face Transformers: Natural language processing models for tasks like text classification, translation, and summarization.
- Ultralytics YOLO: State-of-the-art real-time object detection models.
- Custom Models: Support for your own machine learning models and architectures.

## Prerequisites
Install [PyTorch](https://pytorch.org/get-started/locally/).

## Installation
Install xinfer using pip:
```bash
pip install xinfer
```

Or locally:
```bash
pip install -e .
```

Install PyTorch and transformers in your environment.

## Getting Started

Here's a quick example demonstrating how to use xinfer with a Transformers model:

```python
import xinfer

# Instantiate a Transformers model
model = xinfer.create_model("vikhyatk/moondream2", "transformers")

# Input data
image = "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg"
prompt = "Describe this image. "

# Run inference
output = model.inference(image, prompt, max_new_tokens=50)

print(output)

>>> An animated character with long hair and a serious expression is eating a large burger at a table, with other characters in the background.
```

See [example.ipynb](nbs/example.ipynb) for more examples.


## Supported Models
Transformers:
- BLIP2 Series
- Moondream2


> [!NOTE]
> You can load any [Vision2Seq model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForVision2Seq) 
> from Transformers by using the `Vision2SeqModel` class.

TIMM:
- EVA02 Series

> [!NOTE]
> You can load any model from TIMM by using the `TIMMModel` class.

Ultralytics:
- YOLOv8 Series
- YOLOv10 Series
- YOLOv11 Series

> [!NOTE]
> You can load any model from Ultralytics by using the `UltralyticsModel` class.

Get a list of available models:
```python
import xinfer

xinfer.list_models()
```

<table>
  <thead>
    <tr>
      <th colspan="3" style="text-align: center;">Available Models</th>
    </tr>
    <tr>
      <th>Implementation</th>
      <th>Model ID</th>
      <th>Input --> Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>timm</td>
      <td>eva02_large_patch14_448.mim_m38m_ft_in22k_in1k</td>
      <td>image --> class</td>
    </tr>
    <tr>
      <td>timm</td>
      <td>eva02_large_patch14_448.mim_m38m_ft_in1k</td>
      <td>image --> class</td>
    </tr>
    <tr>
      <td>timm</td>
      <td>eva02_large_patch14_448.mim_in22k_ft_in22k_in1k</td>
      <td>image --> class</td>
    </tr>
    <tr>
      <td>timm</td>
      <td>eva02_large_patch14_448.mim_in22k_ft_in1k</td>
      <td>image --> class</td>
    </tr>
    <tr>
      <td>timm</td>
      <td>eva02_base_patch14_448.mim_in22k_ft_in22k_in1k</td>
      <td>image --> class</td>
    </tr>
    <tr>
      <td>timm</td>
      <td>eva02_base_patch14_448.mim_in22k_ft_in1k</td>
      <td>image --> class</td>
    </tr>
    <tr>
      <td>timm</td>
      <td>eva02_small_patch14_336.mim_in22k_ft_in1k</td>
      <td>image --> class</td>
    </tr>
    <tr>
      <td>timm</td>
      <td>eva02_tiny_patch14_336.mim_in22k_ft_in1k</td>
      <td>image --> class</td>
    </tr>
    <tr>
      <td>transformers</td>
      <td>Salesforce/blip2-opt-6.7b-coco</td>
      <td>image-text --> text</td>
    </tr>
    <tr>
      <td>transformers</td>
      <td>Salesforce/blip2-flan-t5-xxl</td>
      <td>image-text --> text</td>
    </tr>
    <tr>
      <td>transformers</td>
      <td>Salesforce/blip2-opt-6.7b</td>
      <td>image-text --> text</td>
    </tr>
    <tr>
      <td>transformers</td>
      <td>Salesforce/blip2-opt-2.7b</td>
      <td>image-text --> text</td>
    </tr>
    <tr>
      <td>transformers</td>
      <td>vikhyatk/moondream2</td>
      <td>image-text --> text</td>
    </tr>
    <tr>
      <td>ultralytics</td>
      <td>yolov8x</td>
      <td>image --> objects</td>
    </tr>
    <tr>
      <td>ultralytics</td>
      <td>yolov8m</td>
      <td>image --> objects</td>
    </tr>
    <tr>
      <td>ultralytics</td>
      <td>yolov8l</td>
      <td>image --> objects</td>
    </tr>
    <tr>
      <td>ultralytics</td>
      <td>yolov8s</td>
      <td>image --> objects</td>
    </tr>
    <tr>
      <td>ultralytics</td>
      <td>yolov8n</td>
      <td>image --> objects</td>
    </tr>
    <tr>
      <td>ultralytics</td>
      <td>yolov10x</td>
      <td>image --> objects</td>
    </tr>
    <tr>
      <td>ultralytics</td>
      <td>yolov10m</td>
      <td>image --> objects</td>
    </tr>
    <tr>
      <td colspan="3">...</td>
    </tr>
    <tr>
      <td colspan="3">...</td>
    </tr>
  </tbody>
</table>

## Adding New Models

+ Step 1: Create a new model class that implements the `BaseModel` interface.

+ Step 2: Implement the required abstract methods `load_model` and `inference`.

+ Step 3: Decorate your class with the `register_model` decorator, specifying the model ID, backend, and input/output.

