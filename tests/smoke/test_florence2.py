from pathlib import Path

import pytest
import torch

import xinfer

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture
def model():
    return xinfer.create_model(
        "microsoft/Florence-2-base-ft", device="cpu", dtype="float32"
    )


@pytest.fixture
def test_images():
    return [
        str(TEST_DATA_DIR / "test_image_1.jpg"),
        str(TEST_DATA_DIR / "test_image_2.jpg"),
    ]


def test_florence2_initialization(model):
    assert model.model_id == "microsoft/Florence-2-base-ft"
    assert model.device == "cpu"
    assert model.dtype == torch.float32


def test_florence2_inference(model, test_images):
    prompt = "<CAPTION>"
    result = model.infer(test_images[0], prompt)

    assert isinstance(result.text, str)
    assert len(result.text) > 0


def test_florence2_batch_inference(model, test_images):
    prompt = "<CAPTION>"
    result = model.infer_batch(test_images, [prompt, prompt])

    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0].text, str)
    assert isinstance(result[1].text, str)
    assert len(result[0].text) > 0
    assert len(result[1].text) > 0
