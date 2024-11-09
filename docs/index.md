[python_badge]: https://img.shields.io/badge/Python-3.10+-brightgreen?style=for-the-badge&logo=python&logoColor=white
[pypi_badge]: https://img.shields.io/pypi/v/xinfer.svg?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI&color=blue
[downloads_badge]: https://img.shields.io/pepy/dt/xinfer.svg?style=for-the-badge&logo=pypi&logoColor=white&label=Downloads&color=purple
[license_badge]: https://img.shields.io/badge/License-Apache%202.0-green.svg?style=for-the-badge&logo=apache&logoColor=white
[transformers_badge]: https://img.shields.io/github/stars/huggingface/transformers?style=for-the-badge&logo=huggingface&label=Transformers%20⭐&color=yellow
[timm_badge]: https://img.shields.io/github/stars/huggingface/pytorch-image-models?style=for-the-badge&logo=pytorch&label=TIMM%20⭐&color=limegreen
[ultralytics_badge]: https://img.shields.io/github/stars/ultralytics/ultralytics?style=for-the-badge&logo=udacity&label=Ultralytics%20⭐&color=red
[vllm_badge]: https://img.shields.io/github/stars/vllm-project/vllm?style=for-the-badge&logo=v&label=vLLM%20⭐&color=purple
[ollama_badge]: https://img.shields.io/github/stars/ollama/ollama?style=for-the-badge&logo=ollama&label=Ollama%20⭐&color=darkgreen
[colab_badge]: https://img.shields.io/badge/Open%20In-Colab-blue?style=for-the-badge&logo=google-colab
[kaggle_badge]: https://img.shields.io/badge/Open%20In-Kaggle-blue?style=for-the-badge&logo=kaggle
[back_to_top_badge]: https://img.shields.io/badge/Back_to_Top-↑-blue?style=for-the-badge
[image_classification_badge]: https://img.shields.io/badge/Image%20Classification-blueviolet?style=for-the-badge
[object_detection_badge]: https://img.shields.io/badge/Object%20Detection-coral?style=for-the-badge
[image_to_text_badge]: https://img.shields.io/badge/Image%20to%20Text-gold?style=for-the-badge
[os_badge]: https://img.shields.io/badge/Tested%20on-Linux%20%7C%20macOS%20%7C%20Windows-indigo?style=for-the-badge&logo=iterm2&logoColor=white&color=indigo


[![Python][python_badge]](https://pypi.org/project/xinfer/)
[![PyPI version][pypi_badge]](https://pypi.org/project/xinfer/)
[![Downloads][downloads_badge]](https://pypi.org/project/xinfer/)
[![License][license_badge]](https://pypi.org/project/xinfer/)
[![OS Support][os_badge]](https://pypi.org/project/xinfer/)


<div align="center">
    <img src="https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/xinfer.jpg" alt="x.infer" width="500"/>
    <img src="https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/code_typing.gif" alt="x.infer" width="500"/>
    <br />
    <br />
    <a href="https://dnth.github.io/x.infer" target="_blank" rel="noopener noreferrer"><strong>Explore the docs »</strong></a>
    <br />
    <a href="#-quickstart" target="_blank" rel="noopener noreferrer">Quickstart</a>
    ·
    <a href="https://github.com/dnth/x.infer/issues/new?assignees=&labels=Feature+Request&projects=&template=feature_request.md" target="_blank" rel="noopener noreferrer">Feature Request</a>
    ·
    <a href="https://github.com/dnth/x.infer/issues/new?assignees=&labels=bug&projects=&template=bug_report.md" target="_blank" rel="noopener noreferrer">Report Bug</a>
    ·
    <a href="https://github.com/dnth/x.infer/discussions" target="_blank" rel="noopener noreferrer">Discussions</a>
    ·
    <a href="https://dicksonneoh.com/" target="_blank" rel="noopener noreferrer">About</a>
</div>

<div align="center">
    <br />
    
</div>


## 🤔 Why x.infer?
So, a new computer vision model just dropped last night. It's called `GPT-54o-mini-vision-pro-max-xxxl`. It's a super cool model, open-source, open-weights, open-data, all the good stuff.

You're excited. You want to try it out. 

But it's written in a new framework, `TyPorch` that you know nothing about.
You don't want to spend a weekend learning `TyPorch` just to find out the model is not what you expected.

This is where x.infer comes in. 

x.infer is a simple library that allows you to run inference with any computer vision model in just a few lines of code. All in Python.

Out of the box, x.infer supports the following frameworks:

[![Transformers][transformers_badge]](https://github.com/huggingface/transformers)
[![TIMM][timm_badge]](https://github.com/huggingface/pytorch-image-models)
[![Ultralytics][ultralytics_badge]](https://github.com/ultralytics/ultralytics)
[![vLLM][vllm_badge]](https://github.com/vllm-project/vllm)
[![Ollama][ollama_badge]](https://github.com/ollama/ollama)

Combined, x.infer supports over 1000+ models from all the above frameworks.

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

Have a custom model? Create a class that implements the `BaseXInferModel` interface and register it with x.infer. See [Add Your Own Model](#add-your-own-model) for more details.

## 🌟 Key Features
<div align="center">
  <img src="https://raw.githubusercontent.com/dnth/x.infer/refs/heads/main/assets/flowchart.gif" alt="x.infer" width="900"/>
</div>

- **Unified Interface:** Interact with different computer vision frameworks through a single, consistent API.
- **Modular Design:** Integrate and swap out models without altering the core framework.
- **Extensibility:** Add support for new models and libraries with minimal code changes.