from pathlib import Path

import pytest
import torch

from xinfer.transformers.moondream import Moondream


def test_moondream_initialization():
    model = Moondream(device="cpu")
    assert model.model_id == "vikhyatk/moondream2"
    assert model.device == "cpu"
    assert model.dtype == torch.float32


def test_moondream_inference(mocker):
    # Mock the heavy components
    mocker.patch("xinfer.transformers.moondream.AutoModelForCausalLM")
    mocker.patch("xinfer.transformers.moondream.AutoTokenizer")
    mocker.patch("torch.compile", return_value=mocker.MagicMock())

    model = Moondream(device="cpu")

    # Mock the model's specific methods
    model.model.encode_image.return_value = "mock_encoded_image"
    model.model.answer_question.return_value = "This is a test response"

    # Create a test image path
    test_image = str(Path(__file__).parent.parent / "test_data" / "test_image_1.jpg")

    # Test single inference
    result = model.infer(image=test_image, prompt="What's in this image?")
    assert isinstance(result, str)
    assert result == "This is a test response"

    # Test batch inference
    model.model.batch_answer.return_value = ["Response 1", "Response 2"]
    results = model.infer_batch(
        images=[test_image, test_image], prompts=["Question 1", "Question 2"]
    )
    assert isinstance(results, list)
    assert len(results) == 2
    assert results == ["Response 1", "Response 2"]


def test_moondream_invalid_input():
    model = Moondream(device="cpu")

    with pytest.raises(Exception):
        model.infer(image="nonexistent_image.jpg", prompt="What's in this image?")
