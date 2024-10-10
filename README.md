# InferX
A unified interface to run inference on machine learning libraries.


## Overview
InferX is a modular Python framework that provides a unified interface for performing inference across a variety of machine learning models and libraries. Designed to simplify and standardize the inference process, InferX allows developers to work seamlessly with models from Hugging Face Transformers, Ultralytics YOLO, and custom-built models using a consistent and easy-to-use API.

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


## Installation
Install InferX using pip:
```bash
pip install inferx
```

## Getting Started

Here's a quick example demonstrating how to use InferX with a Transformers model:

```python


from inferx import ModelFactory

# Instantiate a Transformers model
model = ModelFactory.get_model(
    model_type='transformers',
    model_name_or_path='distilbert-base-uncased'
)

# Input data
input_text = "Hello, world!"

# Run inference
processed_input = model.preprocess(input_text)
prediction = model.predict(processed_input)
output = model.postprocess(prediction)

print(output)
```
