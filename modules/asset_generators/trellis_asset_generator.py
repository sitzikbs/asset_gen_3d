import os
import logging
from .base_asset_generator import BaseAssetGenerator
from typing import Dict, Any
from PIL import Image

# Import TRELLIS pipeline
from third_party.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from third_party.TRELLIS.trellis.utils import postprocessing_utils

class TrellisAssetGenerator(BaseAssetGenerator):
    def __init__(self, secrets: Dict[str, Any], output_dir: str = "output/assets", model_id: str = "microsoft/TRELLIS-image-large", **kwargs) -> None:
        super().__init__(secrets)
        self.output_dir = output_dir
        self.model_id = model_id
        # Accept and store any additional parameters
        self.extra_params = kwargs
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            logging.info(f"Loading TRELLIS pipeline: {self.model_id}")
            self._pipeline = TrellisImageTo3DPipeline.from_pretrained(self.model_id)
            self._pipeline.cuda()
        return self._pipeline

    def generate_asset(self, image_path: str, prompt_name: str, seed: int = 1) -> str:
        """
        Generates a 3D asset using TRELLIS from an input image and returns the asset path.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        if not image_path or not os.path.exists(image_path):
            logging.warning("TrellisAssetGenerator: No valid image_path provided. Returning None.")
            return None

        pipeline = self._load_pipeline()
        image = Image.open(image_path)
        logging.info(f"TrellisAssetGenerator: Generating 3D asset for '{prompt_name}' using image '{image_path}'.")

        outputs = pipeline.run(image, seed=seed)

        # Save the first mesh as GLB
        glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0])
        asset_filename = f"{prompt_name}.glb"
        asset_path = os.path.join(self.output_dir, asset_filename)
        glb.export(asset_path)
        logging.info(f"--- TRELLIS asset generated at: '{asset_path}' ---")
        return asset_path


if __name__ == "__main__":
    # Use a real image from the assets directory for testing
    print("[TrellisAssetGenerator] Running minimal test with real image...")
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../assets'))
    test_image = os.path.join(assets_dir, 'treasure_chest.png')
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../outputs/test_asset_gen'))
    os.makedirs(output_dir, exist_ok=True)
    seed = 42  # Fixed seed for reproducibility

    try:
        gen = TrellisAssetGenerator(secrets={}, output_dir=output_dir)
        asset_path = gen.generate_asset(test_image, "test_medieval_treasure_chest", seed=seed)
    except Exception as e:
        print(f"[TrellisAssetGenerator] Test failed: {e}")
