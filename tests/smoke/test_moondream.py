from pathlib import Path

import pytest
import torch

import xinfer


@pytest.fixture
def model():
    return xinfer.create_model("vikhyatk/moondream2", device="cpu", dtype="float32")


@pytest.fixture
def test_image():
    return str(Path(__file__).parent.parent / "test_data" / "test_image_1.jpg")


def test_moondream_initialization(model):
    assert model.model_id == "vikhyatk/moondream2"
    assert model.device == "cpu"
    assert model.dtype == torch.float32


def test_moondream_inference(model, test_image):
    prompt = "Caption this image."
    result = model.infer(test_image, prompt)

    assert isinstance(result, str)
    assert len(result) > 0


def test_moondream_batch_inference(model, test_image):
    prompt = "Caption this image."
    result = model.infer_batch([test_image, test_image], [prompt, prompt])

    assert isinstance(result, list)
    assert len(result) == 2
