from .base_text_generator import BaseTextGenerator

class MockTextGenerator(BaseTextGenerator):
    def __init__(self, secrets=None):
        pass

    def generate_prompt(self, base_text):
        """
        Generates a mock detailed prompt.
        """
        return f"A detailed and elaborate description of: {base_text}"
