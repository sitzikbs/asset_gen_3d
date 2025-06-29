from .base_text_generator import BaseTextGenerator
import logging
import os
from typing import Dict, Any

class MockTextGenerator(BaseTextGenerator):
    def __init__(self, secrets: Dict[str, Any], output_dir: str = "output/prompts") -> None:
        super().__init__(secrets, output_dir)
        # Directory creation removed from __init__; will be handled in generate_prompt

    def generate_prompt(self, base_prompt: str, prompt_name: str) -> str:
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"MockTextGenerator: Generating detailed prompt for '{prompt_name}'.")
        # No longer writing a .txt file, as all info is in the main JSON
        return f"mock detailed prompt from: {base_prompt}"
