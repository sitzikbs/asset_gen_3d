import json
import importlib
import argparse
import logging
import os
import shutil
import uuid
from typing import Dict, Any, Optional, List

from modules.text_generators.base_text_generator import BaseTextGenerator
from modules.image_generators.base_image_generator import BaseImageGenerator
from modules.asset_generators.base_asset_generator import BaseAssetGenerator
from utils.file_utils import load_json_config


class Pipeline:
    text_generator: BaseTextGenerator
    image_generator: BaseImageGenerator
    asset_generator: BaseAssetGenerator
    base_output_dir: str
    identifier: str

    def __init__(self, config: Dict[str, Any], secrets: Dict[str, Any], debug: bool = False, identifier: Optional[str] = None) -> None:
        if identifier is None:
            if debug:
                identifier = "debug_run"
            else:
                logging.info("No experiment identifier provided, generating a new unique identifier for this run.")
                identifier = str(uuid.uuid4())
        self.identifier = identifier
        self.base_output_dir = os.path.join("output", identifier)
        if debug:
            logging.info("Running in debug mode. Using mock generators.")
            from modules.text_generators.mock_text_generator import MockTextGenerator
            from modules.image_generators.mock_image_generator import MockImageGenerator
            from modules.asset_generators.mock_asset_generator import MockAssetGenerator
            self.text_generator = MockTextGenerator(secrets={}, output_dir=os.path.join(self.base_output_dir, "prompts"))
            self.image_generator = MockImageGenerator(secrets={}, output_dir=os.path.join(self.base_output_dir, "images"))
            self.asset_generator = MockAssetGenerator(secrets={}, output_dir=os.path.join(self.base_output_dir, "assets"))
        else:
            self.text_generator = self._load_generator('text_generators', config.get('text_generator', {}), secrets, os.path.join(self.base_output_dir, "prompts"))
            self.image_generator = self._load_generator('image_generators', config.get('image_generator', {}), secrets, os.path.join(self.base_output_dir, "images"))
            self.asset_generator = self._load_generator('asset_generators', config.get('asset_generator', {}), secrets, os.path.join(self.base_output_dir, "assets"))

    def _load_generator(self, module_type: str, config: Dict[str, str], secrets: Dict[str, Any], output_dir: str) -> Optional[Any]:
        try:
            if not config or 'module' not in config or 'class' not in config:
                raise KeyError(f"Generator configuration is missing or incomplete for {module_type}")
            module_name = f"modules.{module_type}.{config['module']}"
            logging.debug(f"Attempting to load module: {module_name}")
            module = importlib.import_module(module_name)
            generator_class = getattr(module, config['class'])
            logging.debug(f"Successfully loaded generator class: {config['class']}")
            return generator_class(secrets=secrets, output_dir=output_dir)
        except (ImportError, AttributeError, KeyError) as e:
            logging.error(f"Error loading generator: {e}")
            logging.info(f"Falling back to mock generator for {module_type}.")
            if module_type == 'text_generators':
                from modules.text_generators.mock_text_generator import MockTextGenerator
                return MockTextGenerator(secrets={}, output_dir=output_dir)
            elif module_type == 'image_generators':
                from modules.image_generators.mock_image_generator import MockImageGenerator
                return MockImageGenerator(secrets={}, output_dir=output_dir)
            elif module_type == 'asset_generators':
                from modules.asset_generators.mock_asset_generator import MockAssetGenerator
                return MockAssetGenerator(secrets={}, output_dir=output_dir)
            return None

    def _run_single_prompt(self, base_prompt_text: str, prompt_name: str) -> Dict[str, str]:
        """Runs a single prompt through the pipeline and returns the artifact paths."""
        # 1. Generate a detailed prompt
        detailed_prompt = self.text_generator.generate_prompt(base_prompt_text, prompt_name=prompt_name)
        logging.info(f"Generated detailed prompt: {detailed_prompt}")

        # 2. Generate an image
        generated_image_path = self.image_generator.generate_image(detailed_prompt, prompt_name=prompt_name)
        logging.info(f"Generated image at: {generated_image_path}")

        # 3. Generate the 3D asset
        generated_asset_path = self.asset_generator.generate_asset(generated_image_path, prompt_name=prompt_name)
        logging.info(f"Generated 3D asset at: {generated_asset_path}")

        return {
            "detailed_prompt": detailed_prompt,
            "image_path": generated_image_path,
            "asset_path": generated_asset_path
        }

    def run(self, prompts_data: Dict[str, List[Dict[str, str]]]) -> None:
        prompts_output_dir = os.path.join(self.base_output_dir, "prompts")
        for base_prompt_info in prompts_data['prompts']:
            prompt_name = base_prompt_info['name']
            base_prompt_text = base_prompt_info['text']
            logging.info(f"--- Running pipeline for prompt: {prompt_name} ---")
            artifacts = self._run_single_prompt(base_prompt_text, prompt_name)
            if artifacts.get("asset_path"):
                details_path = os.path.join(prompts_output_dir, f"{prompt_name}.json")
                details_data = {
                    "prompt_name": prompt_name,
                    "base_prompt_text": base_prompt_text,
                    "identifier": self.identifier,
                    "artifacts": artifacts
                }
                with open(details_path, 'w') as f:
                    json.dump(details_data, f, indent=4)
                logging.info(f"Saved prompt details to: {details_path}")
            else:
                logging.error(f"Asset generation failed for prompt: {prompt_name}")
            logging.info(f"--- Pipeline run for {prompt_name} finished ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Asset Generation Pipeline")
    parser.add_argument('--config', type=str, default='configs/pipeline_config.json', help='Path to the pipeline configuration file.')
    parser.add_argument('--secrets', type=str, default='secrets.json', help='Path to the secrets file.')
    parser.add_argument('--prompts', type=str, default='configs/prompts.json', help='Path to the prompts file.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode using mock generators.')
    parser.add_argument('--identifier', type=str, default=None, help='Optional run identifier for output directory.')
    args = parser.parse_args()

    # Configure logging based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # In debug mode, we don't need to load the real configs or secrets
    if args.debug:
        config = {}
        prompts_data = {"prompts": [{"name": "debug_prompt", "text": "A test prompt for debugging."}]}
        secrets = {}
    else:
        config = load_json_config(args.config)
        prompts_data = load_json_config(args.prompts)
        secrets = load_json_config(args.secrets)

    if (config is None or prompts_data is None or secrets is None or not prompts_data.get('prompts')):
        logging.error("Exiting due to configuration or secrets errors.")
        return

    pipeline = Pipeline(config, secrets, args.debug, identifier=args.identifier)
    pipeline.run(prompts_data)


if __name__ == "__main__":
    main()
