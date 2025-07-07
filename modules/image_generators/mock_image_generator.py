import os
import logging
from .base_image_generator import BaseImageGenerator
from typing import Dict, Any

class MockImageGenerator(BaseImageGenerator):
    def __init__(self, secrets: Dict[str, Any], output_dir: str = "output/images") -> None:
        super().__init__(secrets)
        self.output_dir = output_dir

    def to_cpu(self):
        """Mock: No-op for moving to CPU."""
        logging.info("MockImageGenerator: to_cpu() called (no-op).")

    def to_gpu(self):
        """Mock: No-op for moving to GPU."""
        logging.info("MockImageGenerator: to_gpu() called (no-op).")

    def generate_image(self, prompt: str, prompt_name: str) -> str:
        """
        Generates a mock image file and returns its path.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"MockImageGenerator: Generating image for '{prompt_name}'.")
        mock_image_path = os.path.join(self.output_dir, f"{prompt_name}.png")
        
        with open(mock_image_path, "w") as f:
            f.write(f"This is a mock image for prompt: {prompt_name}")
            
        logging.info(f"--- Mock image generated at: '{mock_image_path}' ---")
        return mock_image_path
