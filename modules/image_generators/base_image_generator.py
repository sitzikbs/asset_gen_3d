from abc import ABC, abstractmethod
from typing import Dict, Any
import os

class BaseImageGenerator(ABC):
    @abstractmethod
    def to_cpu(self):
        """Move model/resources to CPU. Must be implemented in subclass."""
        pass

    @abstractmethod
    def to_gpu(self):
        """Move model/resources to GPU. Must be implemented in subclass."""
        pass

    def __init__(self, secrets: Dict[str, Any], output_dir: str = "output/images") -> None:
        self.secrets = secrets
        self.output_dir = output_dir

    @abstractmethod
    def generate_image(self, prompt: str, prompt_name: str) -> str:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        pass
