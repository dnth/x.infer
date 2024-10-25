[python_badge]: https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-brightgreen?style=for-the-badge
[pypi_badge]: https://img.shields.io/pypi/v/xinfer.svg?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=blue
[downloads_badge]: https://img.shields.io/pypi/dm/xinfer.svg?style=for-the-badge&logo=pypi&logoColor=white&label=Downloads&color=purple
[license_badge]: https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge&logo=apache&logoColor=white
[transformers_badge]: https://img.shields.io/badge/Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white
[timm_badge]: https://img.shields.io/badge/TIMM-limegreen?style=for-the-badge&logo=pytorch&logoColor=white
[ultralytics_badge]: https://img.shields.io/badge/Ultralytics-red?style=for-the-badge&logo=udacity&logoColor=white
[vllm_badge]: https://img.shields.io/badge/vLLM-purple?style=for-the-badge&logo=v&logoColor=white
[ollama_badge]: https://img.shields.io/badge/Ollama-darkgreen?style=for-the-badge&logo=llama&logoColor=white
[colab_badge]: https://img.shields.io/badge/Open%20In-Colab-blue?style=for-the-badge&logo=google-colab
[kaggle_badge]: https://img.shields.io/badge/Open%20In-Kaggle-blue?style=for-the-badge&logo=kaggle
[back_to_top_badge]: https://img.shields.io/badge/Back_to_Top-↑-blue?style=for-the-badge
[image_classification_badge]: https://img.shields.io/badge/Image%20Classification-blueviolet?style=for-the-badge
[object_detection_badge]: https://img.shields.io/badge/Object%20Detection-coral?style=for-the-badge
[image_to_text_badge]: https://img.shields.io/badge/Image%20to%20Text-gold?style=for-the-badge

![Python][python_badge]
[![PyPI version][pypi_badge]](https://pypi.org/project/xinfer/)
[![Downloads][downloads_badge]](https://pypi.org/project/xinfer/)
![License][license_badge]


<div align="center">
    <img src="https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/xinfer.jpg" alt="x.infer" width="500"/>
    <br />
    <br />
    <a href="https://dnth.github.io/x.infer" target="_blank" rel="noopener noreferrer"><strong>Explore the docs »</strong></a>
    <br />
    <a href="#quickstart" target="_blank" rel="noopener noreferrer">Quickstart</a>
    ·
    <a href="https://github.com/dnth/x.infer/issues/new?assignees=&labels=Feature+Request&projects=&template=feature_request.md" target="_blank" rel="noopener noreferrer">Feature Request</a>
    ·
    <a href="https://github.com/dnth/x.infer/issues/new?assignees=&labels=bug&projects=&template=bug_report.md" target="_blank" rel="noopener noreferrer">Report Bug</a>
    ·
    <a href="https://github.com/dnth/x.infer/discussions" target="_blank" rel="noopener noreferrer">Discussions</a>
    ·
    <a href="https://dicksonneoh.com/" target="_blank" rel="noopener noreferrer">About</a>
</div>


## 🤔 Why x.infer?
<!-- If you'd like to run many models from different libraries without having to rewrite your inference code, x.infer is for you. It has a simple API and is easy to extend.  -->
So, a new computer vision model just dropped last night. It's called `GPT-54o-mini-vision-pro-max-xxxl`. It's a super cool model, open-source, open-weights, open-data, all the good stuff.

You're excited. You want to try it out. 

But it's written in a new framework, `TyPorch`, that you know nothing about.
You don't want to spend a weekend learning `TyPorch` just to find out the model is not what you expected.

This is where x.infer comes in. It's a simple library that allows you to run inference with any computer vision model in just a few lines of code. All in Python.

Out of the box, x.infer supports the following frameworks:

[![Transformers](https://img.shields.io/badge/Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white)](https://github.com/huggingface/transformers)
[![TIMM](https://img.shields.io/badge/TIMM-limegreen?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/huggingface/pytorch-image-models)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-red?style=for-the-badge&logo=udacity&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![vLLM](https://img.shields.io/badge/vLLM-purple?style=for-the-badge&logo=v&logoColor=white)](https://github.com/vllm-project/vllm)
[![Ollama](https://img.shields.io/badge/Ollama-darkgreen?style=for-the-badge&logo=Ollama&logoColor=white)](https://github.com/ollama/ollama)

In total x.infer already supports over 1000+ models from all the above frameworks.

Tasks supported:

![Image Classification][image_classification_badge]
![Object Detection][object_detection_badge]
![Image to Text][image_to_text_badge]

Run any supported model using the following 4 lines of code:

```python
import xinfer

model = xinfer.create_model("vikhyatk/moondream2")
model.infer(image, prompt)         # Run single inference
model.infer_batch(images, prompts) # Run batch inference
model.launch_gradio()              # Launch Gradio interface
```

Have a custom model? Create a class that implements the `BaseModel` interface and register it with x.infer. See [Adding New Models](#adding-new-models) for more details.

## 🌟 Key Features
<div align="center">
  <img src="https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/flowchart.gif" alt="x.infer" width="900"/>
</div>

- **Unified Interface:** Interact with different machine learning models through a single, consistent API.
- **Modular Design:** Integrate and swap out models without altering the core framework.
- **Extensibility:** Add support for new models and libraries with minimal code changes.

## 🚀 Quickstart

Here's a quick example demonstrating how to use x.infer with a Transformers model:

[![Open In Colab][colab_badge]](https://colab.research.google.com/github/dnth/x.infer/blob/main/nbs/quickstart.ipynb)
[![Open In Kaggle][kaggle_badge]](https://kaggle.com/kernels/welcome?src=https://github.com/dnth/x.infer/blob/main/nbs/quickstart.ipynb)

```python
import xinfer

model = xinfer.create_model("vikhyatk/moondream2")

image = "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg"
prompt = "Describe this image. "

model.infer(image, prompt)

>>> An animated character with long hair and a serious expression is eating a large burger at a table, with other characters in the background.
```

Get a list of models:
```python
xinfer.list_models()
```

```
       Available Models                                      
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┓
┃ Implementation ┃ Model ID                                        ┃ Input --> Output     ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━┩
│ timm           │ eva02_large_patch14_448.mim_m38m_ft_in22k_in1k  │ image --> categories │
│ timm           │ eva02_large_patch14_448.mim_m38m_ft_in1k        │ image --> categories │
│ timm           │ eva02_large_patch14_448.mim_in22k_ft_in22k_in1k │ image --> categories │
│ timm           │ eva02_large_patch14_448.mim_in22k_ft_in1k       │ image --> categories │
│ timm           │ eva02_base_patch14_448.mim_in22k_ft_in22k_in1k  │ image --> categories │
│ timm           │ eva02_base_patch14_448.mim_in22k_ft_in1k        │ image --> categories │
│ timm           │ eva02_small_patch14_336.mim_in22k_ft_in1k       │ image --> categories │
│ timm           │ eva02_tiny_patch14_336.mim_in22k_ft_in1k        │ image --> categories │
│ transformers   │ Salesforce/blip2-opt-6.7b-coco                  │ image-text --> text  │
│ transformers   │ Salesforce/blip2-flan-t5-xxl                    │ image-text --> text  │
│ transformers   │ Salesforce/blip2-opt-6.7b                       │ image-text --> text  │
│ transformers   │ Salesforce/blip2-opt-2.7b                       │ image-text --> text  │
│ transformers   │ fancyfeast/llama-joycaption-alpha-two-hf-llava  │ image-text --> text  │
│ transformers   │ vikhyatk/moondream2                             │ image-text --> text  │
│ transformers   │ sashakunitsyn/vlrm-blip2-opt-2.7b               │ image-text --> text  │
│ ultralytics    │ yolov8x                                         │ image --> boxes      │
│ ultralytics    │ yolov8m                                         │ image --> boxes      │
│ ultralytics    │ yolov8l                                         │ image --> boxes      │
│ ultralytics    │ yolov8s                                         │ image --> boxes      │
│ ultralytics    │ yolov8n                                         │ image --> boxes      │
│ ...            │ ...                                             │ ...                  │
│ ...            │ ...                                             │ ...                  │
└────────────────┴─────────────────────────────────────────────────┴──────────────────────┘
```

## 🖥️ Launch Gradio Interface

For all models, you can launch a Gradio interface to interact with the model.

```python
model.launch_gradio()
```

![Gradio Interface](https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/gradio.png)

You can also launch a Gradio interface with all models available in a dropdown. Pick a model, upload an image, and see the inference results.

```python
xinfer.launch_gradio_demo()
```

![Gradio Demo](https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/gradio_demo.png)

See [Gradio Demo](./nbs/gradio_demo.ipynb) for more details.




## 📦 Installation
> [!IMPORTANT]
> You must have [PyTorch](https://pytorch.org/get-started/locally/) installed to use x.infer.

To install the barebones x.infer (without any optional dependencies), run:
```bash
pip install xinfer
```
x.infer can be used with multiple optional libraries. You'll just need to install one or more of the following:

```bash
pip install "xinfer[transformers]"
pip install "xinfer[ultralytics]"
pip install "xinfer[timm]"
pip install "xinfer[vllm]"
```

To install all libraries, run:
```bash
pip install "xinfer[all]"
```

To install from a local directory, run:
```bash
git clone https://github.com/dnth/x.infer.git
cd x.infer
pip install -e .
```

## 🛠️ Usage


### Supported Models


<details>
<summary><a href="https://github.com/huggingface/transformers"><img src="https://img.shields.io/badge/Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white" alt="Transformers"></a></summary>

<!DOCTYPE html>
<html lang="en">
<body>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Usage</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><a href="https://huggingface.co/Salesforce/blip2-opt-2.7b">BLIP2 Series</a></td>
                <td><code>xinfer.create_model("Salesforce/blip2-opt-2.7b")</code></td>
            </tr>
            <tr>
                <td><a href="https://github.com/vikhyat/moondream">Moondream2</a></td>
                <td><code>xinfer.create_model("vikhyatk/moondream2")</code></td>
            </tr>
            <tr>
                <td><a href="https://huggingface.co/sashakunitsyn/vlrm-blip2-opt-2.7b">VLRM-BLIP2</a></td>
                <td><code>xinfer.create_model("sashakunitsyn/vlrm-blip2-opt-2.7b")</code></td>
            </tr>
            <tr>
                <td><a href="https://github.com/fpgaminer/joycaption">JoyCaption</a></td>
                <td><code>xinfer.create_model("fancyfeast/llama-joycaption-alpha-two-hf-llava")</code></td>
            </tr>
        </tbody>
    </table>
</body>
</html>


You can also load any [Vision2Seq model](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForVision2Seq) 
from Transformers by using the `Vision2SeqModel` class.

```python
from xinfer.transformers import Vision2SeqModel

model = Vision2SeqModel("facebook/chameleon-7b")
model = xinfer.create_model(model)
```

</details>

<details>
<summary><a href="https://github.com/huggingface/pytorch-image-models"><img src="https://img.shields.io/badge/TIMM-green?style=for-the-badge&logo=pytorch&logoColor=white" alt="TIMM"></a></summary>

All models from [TIMM](https://github.com/huggingface/pytorch-image-models) fine-tuned for ImageNet 1k are supported.

For example load a `resnet18.a1_in1k` model:
```python
xinfer.create_model("resnet18.a1_in1k")
```

You can also load any model (or a custom timm model) by using the `TIMMModel` class.

```python
from xinfer.timm import TimmModel

model = TimmModel("resnet18")
model = xinfer.create_model(model)
```

</details>

<details>
<summary><a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/Ultralytics-red?style=for-the-badge&logo=udacity&logoColor=white" alt="Ultralytics"></a></summary>

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Usage</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://github.com/ultralytics/ultralytics">YOLOv8 Series</a></td>
            <td><code>xinfer.create_model("yolov8n")</code></td>
        </tr>
        <tr>
            <td><a href="https://github.com/ultralytics/ultralytics">YOLOv10 Series</a></td>
            <td><code>xinfer.create_model("yolov10x")</code></td>
        </tr>
        <tr>
            <td><a href="https://github.com/ultralytics/ultralytics">YOLOv11 Series</a></td>
            <td><code>xinfer.create_model("yolov11s")</code></td>
        </tr>
    </tbody>
</table>


You can also load any model from Ultralytics by using the `UltralyticsModel` class.

```python
from xinfer.ultralytics import UltralyticsModel

model = UltralyticsModel("yolov5n6u")
model = xinfer.create_model(model)
```

</details>

<details>
<summary><a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/badge/vLLM-purple?style=for-the-badge&logo=v&logoColor=white" alt="vLLM"></a></summary>

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Usage</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://huggingface.co/allenai/Molmo-72B-0924">Molmo-72B</a></td>
            <td><code>xinfer.create_model("allenai/Molmo-72B-0924")</code></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/allenai/Molmo-7B-D-0924">Molmo-7B-D</a></td>
            <td><code>xinfer.create_model("allenai/Molmo-7B-D-0924")</code></td>
        </tr>
        <tr>
            <td><a href="https://huggingface.co/allenai/Molmo-7B-O-0924">Molmo-7B-O</a></td>
            <td><code>xinfer.create_model("allenai/Molmo-7B-O-0924")</code></td>
        </tr>
    </tbody>
</table>

</details>

<details>
<summary><a href="https://github.com/ollama/ollama"><img src="https://img.shields.io/badge/Ollama-darkgreen?style=for-the-badge&logo=Ollama&logoColor=white" alt="Ollama"></a></summary>

<table>
    <thead>
        <tr>
            <th>Model</th>
            <th>Usage</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><a href="https://github.com/ollama/ollama">LLaVA Phi3</a></td>
            <td><code>xinfer.create_model("ollama/llava-phi3")</code></td>
        </tr>
    </tbody>
</table>
</details>

### 🔧 Adding New Models

+ **Step 1:** Create a new model class that implements the `BaseModel` interface.

+ **Step 2:** Implement the required abstract methods `load_model`, `infer`, and `infer_batch`.

+ **Step 3:** Decorate your class with the `register_model` decorator, specifying the model ID, implementation, and input/output.

For example:
```python
@register_model("my-model", "custom", ModelInputOutput.IMAGE_TEXT_TO_TEXT)
class MyModel(BaseModel):
    def load_model(self):
        # Load your model here
        pass

    def infer(self, image, prompt):
        # Run single inference 
        pass

    def infer_batch(self, images, prompts):
        # Run batch inference here
        pass
```

See an example implementation of the Molmo model [here](https://github.com/dnth/x.infer/blob/main/xinfer/vllm/molmo.py).

## 🤝 Contributing

If you'd like to contribute, here are some ways you can help:

1. **Add support for new models:** Implement new model classes following the steps in the [Adding New Models](#-adding-new-models) section.

2. **Improve documentation:** Help us enhance our documentation, including this README, inline code comments, and the [official docs](https://dnth.github.io/x.infer).

3. **Report bugs:** If you find a bug, please [open an issue](https://github.com/dnth/x.infer/issues/new?assignees=&labels=bug&projects=&template=bug_report.md) with a clear description and steps to reproduce.

4. **Suggest enhancements:** Have ideas for new features? [Open a feature request](https://github.com/dnth/x.infer/issues/new?assignees=&labels=Feature+Request&projects=&template=feature_request.md).

5. **Submit pull requests:** Feel free to fork the repository and submit pull requests for any improvements you've made.

Thank you for helping make x.infer better!

<div align="center">
    <img src="https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/github_banner.png" alt="x.infer" width="600"/>
    <br />
    <br />
    <a href="https://dnth.github.io/x.infer" target="_blank" rel="noopener noreferrer"><strong>Explore the docs »</strong></a>
    <br />
    <a href="#quickstart" target="_blank" rel="noopener noreferrer">Quickstart</a>
    ·
    <a href="https://github.com/dnth/x.infer/issues/new?assignees=&labels=Feature+Request&projects=&template=feature_request.md" target="_blank" rel="noopener noreferrer">Feature Request</a>
    ·
    <a href="https://github.com/dnth/x.infer/issues/new?assignees=&labels=bug&projects=&template=bug_report.md" target="_blank" rel="noopener noreferrer">Report Bug</a>
    ·
    <a href="https://github.com/dnth/x.infer/discussions" target="_blank" rel="noopener noreferrer">Discussions</a>
    ·
    <a href="https://dicksonneoh.com/" target="_blank" rel="noopener noreferrer">About</a>
</div>



<div align="right">
    <br />
    <a href="#top"><img src="https://img.shields.io/badge/Back_to_Top-↑-blue?style=for-the-badge" alt="Back to Top" /></a>
</div>








