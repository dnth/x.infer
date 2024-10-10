from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def load_model(self, **kwargs):
        pass

    @abstractmethod
    def preprocess(self, input_data):
        pass

    @abstractmethod
    def predict(self, processed_data):
        pass

    @abstractmethod
    def postprocess(self, prediction):
        pass
