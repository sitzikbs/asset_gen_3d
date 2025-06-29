from abc import ABC, abstractmethod
from typing import Dict, Any
import os

class BaseAssetGenerator(ABC):
    def __init__(self, secrets: Dict[str, Any], output_dir: str = "output/assets") -> None:
        self.secrets = secrets
        self.output_dir = output_dir

    @abstractmethod
    def generate_asset(self, image_path: str, prompt_name: str) -> str:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        pass
