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
│ transformers │ vikhyatk/moondream2               │ image-text --> text │
└──────────────┴───────────────────────────────────┴─────────────────────┘
```

## Loading and Using a Model

You can load and use any of the available models. Here's an example using the Moondream2 model:

```python
# Instantiate a Transformers model
model = xinfer.create_model("vikhyatk/moondream2", backend="transformers")

# Input data
image = "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg"
prompt = "Describe this image."

# Run inference
output = model.inference(image, prompt, max_new_tokens=50)

print(output)
```

This will produce a description of the image, such as:
"An animated character with long hair and a serious expression is eating a large burger at a table, with other characters in the background."

You can use the same pattern for other models like BLIP2 or VLRM-finetuned BLIP2:

```python
# For BLIP2
model = xinfer.create_model("Salesforce/blip2-opt-2.7b", backend="transformers")

# For VLRM-finetuned BLIP2
model = xinfer.create_model("sashakunitsyn/vlrm-blip2-opt-2.7b", backend="transformers")
```

Use the models in the same way as demonstrated with the Moondream2 model.
