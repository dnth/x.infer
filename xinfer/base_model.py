from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id

    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def inference(self, image, prompt):
        pass
