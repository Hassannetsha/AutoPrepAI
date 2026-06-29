from abc import ABC, abstractmethod

class BaseDetectionStrategy(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def detect(self, value) -> bool:
        pass
