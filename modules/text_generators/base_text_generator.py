from abc import ABC, abstractmethod

class BaseTextGenerator(ABC):
    @abstractmethod
    def generate_prompt(self, base_text):
        pass
