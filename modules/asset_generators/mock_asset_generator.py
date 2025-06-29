from .base_asset_generator import BaseAssetGenerator

class MockAssetGenerator(BaseAssetGenerator):
    def __init__(self, secrets=None):
        pass

    def generate_asset(self, image_path):
        """
        Generates a mock asset path.
        """
        print(f"--- Mock asset generation from image: '{image_path}' ---")
        return "path/to/mock/asset.obj"
