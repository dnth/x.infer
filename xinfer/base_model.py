from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def inference(self, image, prompt):
        pass
