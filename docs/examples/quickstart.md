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




    '2.4.0+cu121'



Also let's check if CUDA is available.


```python
torch.cuda.is_available()
```




    True



x.infer relies on various optional dependencies like transformers, ultralytics, timm, etc.
You don't need to install these dependencies if you don't want to. Just install x.infer with the dependencies you want.

For example, if you'd like to use models from the transformers library, you can install the `transformers` extra - `pip install -Uqq "xinfer[transformers]"`

To install all the dependencies, you can run `!pip install -Uqq "xinfer[all]"`

For this example, we'll install all the dependencies.


```python
!pip install -qq "xinfer[all]"
```

Alternatively, if you'd like to install the bleeding edge version of x.infer, uncomment the following line.


```python
# !pip install "git+https://github.com/dnth/x.infer.git#egg=xinfer[all]"
```

It's recommended to restart the kernel once all the dependencies are installed.


```python
from IPython import get_ipython
get_ipython().kernel.do_shutdown(restart=True)
```

Once completed, let's import x.infer, check the version and list all the models available. Specifying `interactive=True` will launch an interactive table in Jupyter Notebooks.


```python
import xinfer

print(xinfer.__version__)
```

    0.2.0



```python
xinfer.list_models(interactive=True)
```

If you'd like to search for a specific model, you can do so by passing in the `search` parameter.


```python
xinfer.list_models(search="moondream")
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">                       Available Models                       </span>
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Implementation </span>┃<span style="font-weight: bold"> Model ID            </span>┃<span style="font-weight: bold"> Input --&gt; Output    </span>┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080"> transformers   </span>│<span style="color: #800080; text-decoration-color: #800080"> vikhyatk/moondream2 </span>│<span style="color: #008000; text-decoration-color: #008000"> image-text --&gt; text </span>│
└────────────────┴─────────────────────┴─────────────────────┘
</pre>



You can pick any model from the list of models available.
Let's create a model from the `vikhyatk/moondream2` model. We can optionally specify the device and dtype. By default, the model is created on the CPU and the dtype is `float32`.

Since we have GPU available, let's create the model on the GPU and use `float16` precision.


```python
model = xinfer.create_model("vikhyatk/moondream2", device="cuda", dtype="float16")
```

    [32m2024-11-01 17:49:20.675[0m | [1mINFO    [0m | [36mxinfer.models[0m:[36m__init__[0m:[36m63[0m - [1mModel: vikhyatk/moondream2[0m
    [32m2024-11-01 17:49:20.676[0m | [1mINFO    [0m | [36mxinfer.models[0m:[36m__init__[0m:[36m64[0m - [1mDevice: cuda[0m
    [32m2024-11-01 17:49:20.676[0m | [1mINFO    [0m | [36mxinfer.models[0m:[36m__init__[0m:[36m65[0m - [1mDtype: float16[0m
    PhiForCausalLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
      - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
      - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
      - If you are not the owner of the model architecture class, please contact the model code owner to update it.


Now that we have the model, let's infer an image.


```python
from PIL import Image
import requests

image_url = "https://raw.githubusercontent.com/dnth/x.infer/main/assets/demo/00aa2580828a9009.jpg"
Image.open(requests.get(image_url, stream=True).raw)

```




    
![png](quickstart_files/quickstart_20_0.png)
    



You can pass in a url or the path to an image file.


```python
image = "https://raw.githubusercontent.com/dnth/x.infer/main/assets/demo/00aa2580828a9009.jpg"
prompt = "Caption this image."

model.infer(image, prompt)
```




    'A parade with a marching band and a flag-bearing figure passes through a town, with spectators lining the street and a church steeple visible in the background.'



If you'd like to generate a longer caption, you can do so by setting the `max_new_tokens` parameter. You can also pass in any generation parameters supported by the `transformers` library.


```python
image = "https://raw.githubusercontent.com/dnth/x.infer/main/assets/demo/00aa2580828a9009.jpg"
prompt = "Caption this image highlighting the focus of the image and the background in detail."

model.infer(image, prompt, max_new_tokens=500)
```




    'The image captures a lively street scene with a parade taking place. A man in a black jacket is walking down the street, carrying a flag, while a group of people are gathered on the sidewalk, watching the parade. In the background, there is a church steeple and a clock tower, adding to the urban setting. The sky is overcast, casting a soft light over the scene.'



If you'd like to see the inference stats, you can do so by calling the `print_stats` method. This might be useful if you're running some sort of benchmark on the inference time.


```python
model.print_stats()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-style: italic">                    Model Info                     </span>
╭───────────────────────────┬─────────────────────╮
│<span style="font-weight: bold"> Attribute                 </span>│<span style="font-weight: bold"> Value               </span>│
├───────────────────────────┼─────────────────────┤
│<span style="color: #008080; text-decoration-color: #008080"> Model ID                  </span>│<span style="color: #800080; text-decoration-color: #800080"> vikhyatk/moondream2 </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Device                    </span>│<span style="color: #800080; text-decoration-color: #800080"> cuda                </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Dtype                     </span>│<span style="color: #800080; text-decoration-color: #800080"> torch.float16       </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Number of Inferences      </span>│<span style="color: #800080; text-decoration-color: #800080"> 2                   </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Total Inference Time (ms) </span>│<span style="color: #800080; text-decoration-color: #800080"> 2652.6134           </span>│
│<span style="color: #008080; text-decoration-color: #008080"> Average Latency (ms)      </span>│<span style="color: #800080; text-decoration-color: #800080"> 1326.3067           </span>│
╰───────────────────────────┴─────────────────────╯
</pre>



Finally, you can also run batch inference. You'll have to pass in a list of images and prompts.


```python
model.infer_batch([image, image], [prompt, prompt])
```




    ['The image captures a lively street scene with a parade taking place. A man in a black jacket is walking down the street, carrying a flag, while a group of people are gathered on the sidewalk, watching the parade. In the background, there is a church steeple and a clock tower, adding to the urban setting. The sky is overcast, casting a soft light over the scene.',
     'The image captures a lively street scene with a parade taking place. A man in a black jacket is walking down the street, carrying a flag, while a group of people are gathered on the sidewalk, watching the parade. In the background, there is a church steeple and a clock tower, adding to the urban setting. The sky is overcast, casting a soft light over the scene.']



For convenience, you can also launch a Gradio interface to interact with the model.


```python
model.launch_gradio()
```

Finally, you can also launch a Gradio interface to interact with all of the models available in x.infer.


```python
xinfer.launch_gradio_demo()
```

If you are done with experimenting and would like to serve the model, you can do so by calling the `serve_model` method. 

This will start a FastAPI server at http://localhost:8000 powered by Ray Serve, allowing you to interact with your model through a REST API.


```python
xinfer.serve_model("vikhyatk/moondream2", device="cuda", dtype="float16", blocking=False)
```

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    2024-11-01 17:50:18,924	INFO worker.py:1807 -- Started a local Ray instance. View the dashboard at [1m[32m127.0.0.1:8265 [39m[22m
    INFO 2024-11-01 17:50:20,677 serve 59754 api.py:277 - Started Serve in namespace "serve".
    INFO 2024-11-01 17:50:20,687 serve 59754 api.py:259 - Connecting to existing Serve app in namespace "serve". New http options will not be applied.
    WARNING 2024-11-01 17:50:20,687 serve 59754 api.py:85 - The new client HTTP config differs from the existing one in the following fields: ['location']. The new HTTP config is ignored.
    [36m(ProxyActor pid=60094)[0m INFO 2024-11-01 17:50:20,660 proxy 192.168.100.60 proxy.py:1191 - Proxy starting on node 59fe892845fc4393c328aca961f67373221805fa2e0aa1dcefb35e23 (HTTP port: 8000).
    [36m(ServeController pid=60096)[0m INFO 2024-11-01 17:50:20,756 controller 60096 deployment_state.py:1604 - Deploying new version of Deployment(name='XInferModel', app='default') (initial target replicas: 1).
    [36m(ServeController pid=60096)[0m INFO 2024-11-01 17:50:20,861 controller 60096 deployment_state.py:1850 - Adding 1 replica to Deployment(name='XInferModel', app='default').
    [36m(ServeReplica:default:XInferModel pid=60091)[0m 2024-11-01 17:50:24.936 | INFO     | xinfer.models:__init__:63 - Model: vikhyatk/moondream2
    [36m(ServeReplica:default:XInferModel pid=60091)[0m 2024-11-01 17:50:24.936 | INFO     | xinfer.models:__init__:64 - Device: cuda
    [36m(ServeReplica:default:XInferModel pid=60091)[0m 2024-11-01 17:50:24.936 | INFO     | xinfer.models:__init__:65 - Dtype: float16
    [36m(ServeReplica:default:XInferModel pid=60091)[0m PhiForCausalLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
    [36m(ServeReplica:default:XInferModel pid=60091)[0m   - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
    [36m(ServeReplica:default:XInferModel pid=60091)[0m   - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
    [36m(ServeReplica:default:XInferModel pid=60091)[0m   - If you are not the owner of the model architecture class, please contact the model code owner to update it.
    INFO 2024-11-01 17:50:31,785 serve 59754 client.py:492 - Deployment 'XInferModel:0fuqikbq' is ready at `http://127.0.0.1:8000/`. component=serve deployment=XInferModel
    INFO 2024-11-01 17:50:31,790 serve 59754 api.py:549 - Deployed app 'default' successfully.
    [32m2024-11-01 17:50:31.792[0m | [1mINFO    [0m | [36mxinfer.serve[0m:[36mserve_model[0m:[36m89[0m - [1mRunning server in non-blocking mode, remember to call serve.shutdown() to stop the server[0m





    DeploymentHandle(deployment='XInferModel')



Now you can make requests to the model using the python `requests` library or `curl` command.

Using the python `requests` library:


```python
import requests

url = "http://127.0.0.1:8000/infer"
headers = {
    "accept": "application/json",
    "Content-Type": "application/json"
}
payload = {
    "image": "https://raw.githubusercontent.com/dnth/x.infer/main/assets/demo/00aa2580828a9009.jpg",
    "infer_kwargs": {
        "prompt": "Caption this image"
    }
}

response = requests.post(url, headers=headers, json=payload)
print(response.json())
```

    {'response': 'A parade with a marching band and a flag-bearing figure passes through a town, with spectators lining the street and a church steeple visible in the background.'}


Or using the `curl` command:


```bash
%%bash

curl -X 'POST' \
  'http://127.0.0.1:8000/infer' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "image": "https://raw.githubusercontent.com/dnth/x.infer/main/assets/demo/00aa2580828a9009.jpg",
  "infer_kwargs": {"prompt": "Caption this image"}
}'
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   339  100   175  100   164    202    189 --:--:-- --:--:-- --:--:--   392


    [36m(ServeReplica:default:XInferModel pid=60091)[0m INFO 2024-11-01 17:50:32,801 default_XInferModel f3fw2uje e03aff59-5099-4fca-9481-16139b9377fe /infer replica.py:378 - __CALL__ OK 991.6ms
    [36m(ServeReplica:default:XInferModel pid=60091)[0m INFO 2024-11-01 17:50:33,675 default_XInferModel f3fw2uje 4683161e-af95-4bd2-8de9-c1b7fdb7e0fc /infer replica.py:378 - __CALL__ OK 860.1ms
    [36m(ServeController pid=60096)[0m INFO 2024-11-01 17:50:41,391 controller 60096 deployment_state.py:1866 - Removing 1 replica from Deployment(name='XInferModel', app='default').
    [36m(ServeController pid=60096)[0m INFO 2024-11-01 17:50:43,409 controller 60096 deployment_state.py:2191 - Replica(id='f3fw2uje', deployment='XInferModel', app='default') is stopped.


    {"response":"A parade with a marching band and a flag-bearing figure passes through a town, with spectators lining the street and a church steeple visible in the background."}

Since we are using the non-blocking parameter in `serve_model`, we need to shut down the server manually.


```python
from ray import serve

serve.shutdown()
```

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
