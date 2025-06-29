import os
import logging
from .base_asset_generator import BaseAssetGenerator
from typing import Dict, Any

class MockAssetGenerator(BaseAssetGenerator):
    def __init__(self, secrets: Dict[str, Any], output_dir: str = "output/assets") -> None:
        super().__init__(secrets)
        self.output_dir = output_dir

    def generate_asset(self, image_path: str, prompt_name: str) -> str:
        """
        Generates a mock asset file and returns its path.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if not image_path:
            logging.warning("generate_asset called with no image_path. Returning None.")
            return None

        logging.info(f"MockAssetGenerator: Generating 3D asset for '{prompt_name}'.")
        asset_filename = f"{prompt_name}.glb"
        asset_path = os.path.join(self.output_dir, asset_filename)

        with open(asset_path, 'w') as f:
            f.write(f"This is a mock 3D asset for prompt: {prompt_name}")

        logging.info(f"--- Mock asset generated at: '{asset_path}' ---")
        return asset_path
