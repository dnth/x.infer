from pathlib import Path

import pytest
import torch

import xinfer

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture
def model():
    return xinfer.create_model("vikhyatk/moondream2", device="cpu", dtype="float32")


@pytest.fixture
def test_images():
    return [
        str(TEST_DATA_DIR / "test_image_2.jpg"),
        str(TEST_DATA_DIR / "test_image_3.jpg"),
    ]


@pytest.fixture
def test_prompts():
    return ["Caption this image.", "Caption this image."]


def test_moondream_initialization(model):
    assert model.model_id == "vikhyatk/moondream2"
    assert model.device == "cpu"
    assert model.dtype == torch.float32


def test_moondream_inference(model, test_images, test_prompts):
    result = model.infer(test_images[0], test_prompts[0])

    assert isinstance(result.text, str)
    assert len(result.text) > 0


def test_moondream_batch_inference(model, test_images, test_prompts):
    results = model.infer_batch(test_images, test_prompts)

    assert isinstance(results, list)
    assert len(results) == 2
    assert isinstance(results[0].text, str)
    assert isinstance(results[1].text, str)
    assert len(results[0].text) > 0
    assert len(results[1].text) > 0
