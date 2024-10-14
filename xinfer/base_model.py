from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model_id: str):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def infer(self, image: str, prompt: str):
        pass

    @abstractmethod
    def infer_batch(self, images: list[str], prompts: list[str]):
        pass
