from pathlib import Path

import pytest

import xinfer

TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"


@pytest.fixture
def model():
    return xinfer.create_model("timm/resnet18.a1_in1k")


@pytest.fixture
def test_images():
    return [
        str(TEST_DATA_DIR / "test_image_1.jpg"),
        str(TEST_DATA_DIR / "test_image_2.jpg"),
    ]


def test_single_image_inference(model):
    result = model.infer(str(TEST_DATA_DIR / "test_image_2.jpg"), top_k=10)

    assert len(result.categories) == 10
    assert isinstance(result.categories[0].label, str)
    assert isinstance(result.categories[0].score, float)


def test_batch_inference(model, test_images):
    results = model.infer_batch(test_images, top_k=10)

    assert len(results) == 2
    assert len(results[0].categories) == 10
    assert len(results[1].categories) == 10
    assert isinstance(results[0].categories[0].label, str)
    assert isinstance(results[0].categories[0].score, float)
