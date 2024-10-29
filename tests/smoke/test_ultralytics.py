from pathlib import Path

import pytest
import torch

import xinfer


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
    # Test if there is boxes and scores in the first element of the result
    result = model.infer(test_image)[0]

    assert isinstance(result, dict)
    assert "bbox" in result
    assert "score" in result
    assert "class_name" in result
    assert "category_id" in result

    # Test bbox format and values
    assert isinstance(result["bbox"], list)
    assert len(result["bbox"]) == 4  # [x, y, width, height]
    assert all(isinstance(coord, float) for coord in result["bbox"])
    assert all(coord >= 0 for coord in result["bbox"])


def test_ultralytics_batch_inference(model, test_image):
    result = model.infer_batch([test_image, test_image])

    assert isinstance(result, list)
    assert len(result) == 2

    # Verify structure of each batch result
    for batch_result in result:
        assert isinstance(batch_result, list)
        # Check each detection in the batch
        for detection in batch_result:
            assert isinstance(detection, dict)
            assert "bbox" in detection
            assert "score" in detection
            assert "class_name" in detection
            assert "category_id" in detection

            # Verify data types and value ranges
            assert isinstance(detection["bbox"], list)
            assert len(detection["bbox"]) == 4  # [x, y, width, height]
            assert isinstance(detection["score"], float)
            assert 0 <= detection["score"] <= 1  # Score should be between 0 and 1
            assert isinstance(detection["class_name"], str)
            assert isinstance(detection["category_id"], int)
