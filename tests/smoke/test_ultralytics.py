from pathlib import Path

import pytest
import torch

import xinfer
from xinfer.types import Box, Result


@pytest.fixture
def model():
    return xinfer.create_model("ultralytics/yolov8n", device="cpu", dtype="float32")


@pytest.fixture
def test_image():
    return str(Path(__file__).parent.parent / "test_data" / "test_image_1.jpg")


def test_ultralytics_initialization(model):
    assert model.model_id == "ultralytics/yolov8n"
    assert model.device == "cpu"
    assert model.dtype == torch.float32


def test_ultralytics_inference(model, test_image):
    result = model.infer(test_image)

    assert isinstance(result, Result)
    assert result.boxes is not None
    assert len(result.boxes) > 0

    # Test first box format and values
    box = result.boxes[0]
    assert isinstance(box, Box)
    assert isinstance(box.x1, float)
    assert isinstance(box.y1, float)
    assert isinstance(box.x2, float)
    assert isinstance(box.y2, float)
    assert isinstance(box.score, float)
    assert isinstance(box.label, str)
    assert 0 <= box.score <= 1
    assert all(coord >= 0 for coord in [box.x1, box.y1, box.x2, box.y2])


def test_ultralytics_batch_inference(model, test_image):
    results = model.infer_batch([test_image, test_image])

    assert isinstance(results, list)
    assert len(results) == 2

    # Verify structure of each batch result
    for batch_result in results:
        assert isinstance(batch_result, Result)
        assert batch_result.boxes is not None

        # Check each detection in the batch
        for box in batch_result.boxes:
            assert isinstance(box, Box)
            assert isinstance(box.score, float)
            assert 0 <= box.score <= 1
            assert isinstance(box.label, str)
            assert all(
                isinstance(coord, float) for coord in [box.x1, box.y1, box.x2, box.y2]
            )
