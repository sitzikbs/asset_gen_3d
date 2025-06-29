from .base_image_generator import BaseImageGenerator

class MockImageGenerator(BaseImageGenerator):
    def __init__(self, secrets=None):
        pass

    def generate_image(self, prompt):
        """
        Generates a mock image path.
        """
        print(f"--- Mock image generation for prompt: '{prompt}' ---")
        return "path/to/mock/image.png"
