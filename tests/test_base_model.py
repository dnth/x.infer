import pytest

from xinfer.models import BaseModel


class MockModel(BaseModel):
    def load_model(self):
        pass

    def infer(self, image: str, prompt: str):
        pass

    def infer_batch(self, images: list[str], prompts: list[str]):
        pass


@pytest.fixture
def base_model():
    return MockModel("test_model", "cpu", "float32")


def test_base_model_init(base_model):
    assert base_model.model_id == "test_model"
    assert base_model.device == "cpu"
    assert base_model.dtype == "float32"
