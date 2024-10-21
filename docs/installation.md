# Installation

You must have [PyTorch](https://pytorch.org/get-started/locally/) installed to use x.infer.

To install the barebones x.infer (without any optional dependencies), run:
```bash
pip install xinfer
```
x.infer can be used with multiple optional libraries. You'll just need to install one or more of the following:

```bash
pip install "xinfer[transformers]"
pip install "xinfer[ultralytics]"
pip install "xinfer[timm]"
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