import json
import importlib
import argparse
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Pipeline:
    def __init__(self, config, secrets):
        self.text_generator = self._load_generator('text_generators', config['text_generator'], secrets)
        self.image_generator = self._load_generator('image_generators', config['image_generator'], secrets)
        self.asset_generator = self._load_generator('asset_generators', config['asset_generator'], secrets)

    def _load_generator(self, module_type, config, secrets):
        try:
            module_name = f"modules.{module_type}.{config['module']}"
            module = importlib.import_module(module_name)
            generator_class = getattr(module, config['class'])
            # Pass secrets to the generator's constructor
            return generator_class(secrets=secrets)
        except (ImportError, AttributeError) as e:
            logging.error(f"Error loading generator: {e}")
            logging.info(f"Falling back to mock generator for {module_type}.")
            # Mocks don't need secrets, so we pass an empty dict
            if module_type == 'text_generators':
                from modules.text_generators.mock_text_generator import MockTextGenerator
                return MockTextGenerator(secrets={})
            elif module_type == 'image_generators':
                from modules.image_generators.mock_image_generator import MockImageGenerator
                return MockImageGenerator(secrets={})
            elif module_type == 'asset_generators':
                from modules.asset_generators.mock_asset_generator import MockAssetGenerator
                return MockAssetGenerator(secrets={})
            return None # Should not be reached


    def run(self, base_prompt_text):
        # 1. Generate a detailed prompt
        detailed_prompt = self.text_generator.generate_prompt(base_prompt_text)
        logging.info(f"Generated detailed prompt: {detailed_prompt}")

        # 2. Generate an image
        generated_image = self.image_generator.generate_image(detailed_prompt)
        logging.info(f"Generated image at: {generated_image}")

        # 3. Generate the 3D asset
        generated_asset = self.asset_generator.generate_asset(generated_image)
        logging.info(f"Generated 3D asset at: {generated_asset}")

        return generated_asset


def load_json_config(file_path):
    """Loads a JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        return None


def run_pipeline_for_prompts(pipeline, prompts_data):
    """Runs the asset generation pipeline for all prompts."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    for base_prompt_info in prompts_data['prompts']:
        prompt_name = base_prompt_info['name']
        base_prompt_text = base_prompt_info['text']
        logging.info(f"--- Running pipeline for prompt: {prompt_name} ---")
        
        final_asset_path = pipeline.run(base_prompt_text)

        if final_asset_path:
            prompt_output_dir = os.path.join(output_dir, prompt_name)
            os.makedirs(prompt_output_dir, exist_ok=True)
            # In a real implementation, you would copy the file from final_asset_path
            # For this example, we'll create a placeholder file.
            final_destination = os.path.join(prompt_output_dir, os.path.basename(final_asset_path))
            with open(final_destination, 'w') as f:
                f.write(f"This is the generated asset for prompt '{prompt_name}'.\n")
            logging.info(f"Saved final asset to: {final_destination}")

        logging.info(f"--- Pipeline run for {prompt_name} finished ---")


def main():
    parser = argparse.ArgumentParser(description="3D Asset Generation Pipeline")
    parser.add_argument('--config', type=str, default='configs/pipeline_config.json', help='Path to the pipeline configuration file.')
    parser.add_argument('--secrets', type=str, default='secrets.json', help='Path to the secrets file.')
    args = parser.parse_args()

    config = load_json_config(args.config)
    prompts_data = load_json_config('configs/prompts.json')
    secrets = load_json_config(args.secrets)

    if not config or not prompts_data or not secrets:
        logging.error("Exiting due to configuration or secrets errors.")
        return

    pipeline = Pipeline(config, secrets)
    run_pipeline_for_prompts(pipeline, prompts_data)


if __name__ == "__main__":
    main()
