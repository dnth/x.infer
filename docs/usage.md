# Usage

To use xinfer in a project:

```python
import xinfer
```

## Listing Available Models

You can list the available models using the `list_models()` function:

```python
xinfer.list_models()
```

This will display a table of available models and their backends and input/output types.

```
                             Available Models                             
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Backend      ┃ Model ID                          ┃ Input/Output        ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ transformers │ Salesforce/blip2-opt-2.7b         │ image-text --> text │
│ transformers │ sashakunitsyn/vlrm-blip2-opt-2.7b │ image-text --> text │
└──────────────┴───────────────────────────────────┴─────────────────────┘
```

## Loading and Using a Model

### BLIP2 Model

Here's an example of how to load and use the BLIP2 model:

```python
# Instantiate a Transformers model
model = xinfer.create_model("Salesforce/blip2-opt-2.7b", backend="transformers")

# Input data
image = "https://example.com/path/to/image.jpg"
prompt = "What's in this image? Answer:"

# Run inference
processed_input = model.preprocess(image, prompt)
prediction = model.predict(processed_input)
output = model.postprocess(prediction)

print(output)
```

You can also customize the generation parameters:

```python
prediction = model.predict(processed_input, max_new_tokens=200)
```

### VLRM-finetuned BLIP2 Model

Similarly, you can use the VLRM-finetuned BLIP2 model:

```python
model = xinfer.create_model("sashakunitsyn/vlrm-blip2-opt-2.7b", backend="transformers")

# Use the model in the same way as the BLIP2 model
```

Both models can be used for tasks like image description and visual question answering.
