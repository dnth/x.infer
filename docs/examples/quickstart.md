## Quickstart

[![Open In Colab](https://img.shields.io/badge/Open%20In-Colab-blue?style=for-the-badge&logo=google-colab)](https://colab.research.google.com/github/dnth/x.infer/blob/main/nbs/quickstart.ipynb)
[![Open In Kaggle](https://img.shields.io/badge/Open%20In-Kaggle-blue?style=for-the-badge&logo=kaggle)](https://kaggle.com/kernels/welcome?src=https://github.com/dnth/x.infer/blob/main/nbs/quickstart.ipynb)

This notebook shows how to get started with using x.infer.

x.infer relies on PyTorch and torchvision, so make sure you have it installed on your system. Uncomment the following line to install it.


```python
# !pip install -Uqq torch torchvision
```

Let's check if PyTorch is installed by checking the version.


```python
import torch

torch.__version__
```




    '2.2.0+cu121'



Also let's check if CUDA is available.


```python
torch.cuda.is_available()
```




    True



x.infer relies on various optional dependencies like transformers, ultralytics, timm, etc.
You don't need to install these dependencies if you don't want to. Just install x.infer with the dependencies you want.

For example, if you'd like to use models from the transformers library, you can install the `transformers` extra:


```bash
pip install -Uqq "xinfer[transformers]"
```

To install all the dependencies, you can run:
```bash
pip install -Uqq "xinfer[all]"
```

For this example, we'll install all the dependencies.


```python
!pip install -Uqq "xinfer[all]"
```

It's recommended to restart the kernel once all the dependencies are installed. Uncomment the following line to restart the kernel.


```python
# from IPython import get_ipython
# get_ipython().kernel.do_shutdown(restart=True)
```

Once completed, let's import x.infer, check the version and list all the models available.


```python
import xinfer

print(xinfer.__version__)
xinfer.list_models()
```

    0.0.7



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">                                     Available Models                                     </span>
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Implementation </span>┃<span style="font-weight: bold"> Model ID                                        </span>┃<span style="font-weight: bold"> Input --&gt; Output    </span>┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080"> timm           </span>│<span style="color: #800080; text-decoration-color: #800080"> eva02_large_patch14_448.mim_m38m_ft_in22k_in1k  </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; class     </span>│
│<span style="color: #008080; text-decoration-color: #008080"> timm           </span>│<span style="color: #800080; text-decoration-color: #800080"> eva02_large_patch14_448.mim_m38m_ft_in1k        </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; class     </span>│
│<span style="color: #008080; text-decoration-color: #008080"> timm           </span>│<span style="color: #800080; text-decoration-color: #800080"> eva02_large_patch14_448.mim_in22k_ft_in22k_in1k </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; class     </span>│
│<span style="color: #008080; text-decoration-color: #008080"> timm           </span>│<span style="color: #800080; text-decoration-color: #800080"> eva02_large_patch14_448.mim_in22k_ft_in1k       </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; class     </span>│
│<span style="color: #008080; text-decoration-color: #008080"> timm           </span>│<span style="color: #800080; text-decoration-color: #800080"> eva02_base_patch14_448.mim_in22k_ft_in22k_in1k  </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; class     </span>│
│<span style="color: #008080; text-decoration-color: #008080"> timm           </span>│<span style="color: #800080; text-decoration-color: #800080"> eva02_base_patch14_448.mim_in22k_ft_in1k        </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; class     </span>│
│<span style="color: #008080; text-decoration-color: #008080"> timm           </span>│<span style="color: #800080; text-decoration-color: #800080"> eva02_small_patch14_336.mim_in22k_ft_in1k       </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; class     </span>│
│<span style="color: #008080; text-decoration-color: #008080"> timm           </span>│<span style="color: #800080; text-decoration-color: #800080"> eva02_tiny_patch14_336.mim_in22k_ft_in1k        </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; class     </span>│
│<span style="color: #008080; text-decoration-color: #008080"> transformers   </span>│<span style="color: #800080; text-decoration-color: #800080"> Salesforce/blip2-opt-6.7b-coco                  </span>│<span style="color: #008000; text-decoration-color: #008000"> image-text --&gt; text </span>│
│<span style="color: #008080; text-decoration-color: #008080"> transformers   </span>│<span style="color: #800080; text-decoration-color: #800080"> Salesforce/blip2-flan-t5-xxl                    </span>│<span style="color: #008000; text-decoration-color: #008000"> image-text --&gt; text </span>│
│<span style="color: #008080; text-decoration-color: #008080"> transformers   </span>│<span style="color: #800080; text-decoration-color: #800080"> Salesforce/blip2-opt-6.7b                       </span>│<span style="color: #008000; text-decoration-color: #008000"> image-text --&gt; text </span>│
│<span style="color: #008080; text-decoration-color: #008080"> transformers   </span>│<span style="color: #800080; text-decoration-color: #800080"> Salesforce/blip2-opt-2.7b                       </span>│<span style="color: #008000; text-decoration-color: #008000"> image-text --&gt; text </span>│
│<span style="color: #008080; text-decoration-color: #008080"> transformers   </span>│<span style="color: #800080; text-decoration-color: #800080"> fancyfeast/llama-joycaption-alpha-two-hf-llava  </span>│<span style="color: #008000; text-decoration-color: #008000"> image-text --&gt; text </span>│
│<span style="color: #008080; text-decoration-color: #008080"> transformers   </span>│<span style="color: #800080; text-decoration-color: #800080"> vikhyatk/moondream2                             </span>│<span style="color: #008000; text-decoration-color: #008000"> image-text --&gt; text </span>│
│<span style="color: #008080; text-decoration-color: #008080"> transformers   </span>│<span style="color: #800080; text-decoration-color: #800080"> sashakunitsyn/vlrm-blip2-opt-2.7b               </span>│<span style="color: #008000; text-decoration-color: #008000"> image-text --&gt; text </span>│
│<span style="color: #008080; text-decoration-color: #008080"> ultralytics    </span>│<span style="color: #800080; text-decoration-color: #800080"> yolov8x                                         </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; objects   </span>│
│<span style="color: #008080; text-decoration-color: #008080"> ultralytics    </span>│<span style="color: #800080; text-decoration-color: #800080"> yolov8m                                         </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; objects   </span>│
│<span style="color: #008080; text-decoration-color: #008080"> ultralytics    </span>│<span style="color: #800080; text-decoration-color: #800080"> yolov8l                                         </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; objects   </span>│
│<span style="color: #008080; text-decoration-color: #008080"> ultralytics    </span>│<span style="color: #800080; text-decoration-color: #800080"> yolov8s                                         </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; objects   </span>│
│<span style="color: #008080; text-decoration-color: #008080"> ultralytics    </span>│<span style="color: #800080; text-decoration-color: #800080"> yolov8n                                         </span>│<span style="color: #008000; text-decoration-color: #008000"> image --&gt; objects   </span>│
│<span style="color: #008080; text-decoration-color: #008080"> ...            </span>│<span style="color: #800080; text-decoration-color: #800080"> ...                                             </span>│<span style="color: #008000; text-decoration-color: #008000"> ...                 </span>│
│<span style="color: #008080; text-decoration-color: #008080"> ...            </span>│<span style="color: #800080; text-decoration-color: #800080"> ...                                             </span>│<span style="color: #008000; text-decoration-color: #008000"> ...                 </span>│
└────────────────┴─────────────────────────────────────────────────┴─────────────────────┘
</pre>



You can pick any model from the list of models available.
Let's create a model from the `vikhyatk/moondream2` model. We can optionally specify the device and dtype. By default, the model is created on the CPU and the dtype is `float32`.

Since we have GPU available, let's create the model on the GPU and use `float16` precision.


```python
model = xinfer.create_model("vikhyatk/moondream2", device="cuda", dtype="float16")
```

    [32m2024-10-21 22:37:07.833[0m | [1mINFO    [0m | [36mxinfer.transformers.moondream[0m:[36m__init__[0m:[36m24[0m - [1mModel: vikhyatk/moondream2[0m
    [32m2024-10-21 22:37:07.835[0m | [1mINFO    [0m | [36mxinfer.transformers.moondream[0m:[36m__init__[0m:[36m25[0m - [1mRevision: 2024-08-26[0m
    [32m2024-10-21 22:37:07.835[0m | [1mINFO    [0m | [36mxinfer.transformers.moondream[0m:[36m__init__[0m:[36m26[0m - [1mDevice: cuda[0m
    [32m2024-10-21 22:37:07.835[0m | [1mINFO    [0m | [36mxinfer.transformers.moondream[0m:[36m__init__[0m:[36m27[0m - [1mDtype: float16[0m
    PhiForCausalLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
      - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
      - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
      - If you are not the owner of the model architecture class, please contact the model code owner to update it.


Now that we have the model, let's infer an image.


```python
from PIL import Image
import requests

image_url = "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg"
Image.open(requests.get(image_url, stream=True).raw)

```




    
![png](quickstart_files/quickstart_15_0.png)
    



You can pass in a url or the path to an image file.


```python
image = "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg"
prompt = "Caption this image."

model.infer(image, prompt)
```




    'An anime-style illustration depicts a young girl with white hair and green eyes, wearing a white jacket, holding a large burger in her hands and smiling.'



If you'd like to generate a longer caption, you can do so by setting the `max_new_tokens` parameter. You can also pass in any generation parameters supported by the `transformers` library.


```python
image = "https://raw.githubusercontent.com/vikhyat/moondream/main/assets/demo-1.jpg"
prompt = "Caption this image highlighting the focus of the image and the background in detail."

model.infer(image, prompt, max_new_tokens=500)
```




    'The image depicts a young girl with long, white hair and blue eyes sitting at a table, holding a large burger in her hands. The background shows a cozy indoor setting with a window and a chair visible.'



If you'd like to see the inference stats, you can do so by calling the `print_stats` method. This might be useful if you're running some sort of benchmark on the inference time.


```python
model.stats.print_stats()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">                    Model Stats                    </span>
╭───────────────────────────┬─────────────────────╮
│<span style="font-weight: bold"> Attribute                 </span>│<span style="font-weight: bold"> Value               </span>│
├───────────────────────────┼─────────────────────┤
│<span style="color: #008080; text-decoration-color: #008080"> Model ID                  </span>│<span style="color: #800080; text-decoration-color: #800080"> vikhyatk/moondream2 </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Device                    </span>│<span style="color: #800080; text-decoration-color: #800080"> cuda                </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Dtype                     </span>│<span style="color: #800080; text-decoration-color: #800080"> torch.float16       </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Number of Inferences      </span>│<span style="color: #800080; text-decoration-color: #800080"> 2                   </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Total Inference Time (ms) </span>│<span style="color: #800080; text-decoration-color: #800080"> 2087.3517           </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Average Latency (ms)      </span>│<span style="color: #800080; text-decoration-color: #800080"> 1043.6759           </span>│
╰───────────────────────────┴─────────────────────╯
</pre>



Finally, you can also run batch inference. You'll have to pass in a list of images and prompts.


```python
model.infer_batch([image, image], [prompt, prompt])
```




    ['The image depicts a young girl with long, white hair and blue eyes sitting at a table, holding a large burger in her hands. The background shows a cozy indoor setting with a window and a chair visible.',
     'The image depicts a young girl with long, white hair and blue eyes sitting at a table, holding a large burger in her hands. The background shows a cozy indoor setting with a window and a chair visible.']



For convenience, you can also launch a Gradio interface to interact with the model.


```python
model.launch_gradio()
```

    * Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>


That's it! You've successfully run inference with x.infer. 

Hope this simplifies the process of running inference with your favorite computer vision models!

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
