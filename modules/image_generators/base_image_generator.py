from abc import ABC, abstractmethod

class BaseImageGenerator(ABC):
    @abstractmethod
    def generate_image(self, prompt):
        pass
