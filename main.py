import json
import importlib
import argparse
import logging
import os
import uuid
from typing import Dict, Any, Optional, List

from modules.text_generators.base_text_generator import BaseTextGenerator
from modules.image_generators.base_image_generator import BaseImageGenerator
from modules.asset_generators.base_asset_generator import BaseAssetGenerator
from utils.file_utils import load_json_config
from utils.pipeline_utils import get_identifier, configure_logging, get_configs, validate_configs


class Pipeline:
    """
    Orchestrates the 3D asset generation pipeline using modular generator classes.
    Handles configuration, output directory structure, and the full prompt-to-asset workflow.
    """

    text_generator: BaseTextGenerator
    image_generator: BaseImageGenerator
    asset_generator: BaseAssetGenerator
    base_output_dir: str
    identifier: str

    def __init__(self, config: Dict[str, Any], secrets: Dict[str, Any], debug: bool = False, identifier: Optional[str] = None) -> None:
        """
        Initializes the pipeline, sets up output directories, and loads generator classes.
        """
        self.identifier = get_identifier(debug, identifier)
        self.base_output_dir = os.path.join("outputs", self.identifier)
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
        """
        Dynamically loads and instantiates a generator class based on config.
        Passes any 'params' dict as kwargs to the generator constructor.
        Falls back to a mock generator if loading fails.
        """
        try:
            if not config or 'module' not in config or 'class' not in config:
                raise KeyError(f"Generator configuration is missing or incomplete for {module_type}")
            module_name = f"modules.{module_type}.{config['module']}"
            logging.debug(f"Attempting to load module: {module_name}")
            module = importlib.import_module(module_name)
            generator_class = getattr(module, config['class'])
            logging.debug(f"Successfully loaded generator class: {config['class']}")
            params = config.get('params', {})
            return generator_class(secrets=secrets, output_dir=output_dir, **params)
        except (ImportError, AttributeError, KeyError, TypeError) as e:
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

        # 2. Generate an image (defer asset generation)
        generated_image_path = self.image_generator.generate_image(detailed_prompt, prompt_name=prompt_name)
        logging.info(f"Generated image at: {generated_image_path}")

        return {
            "detailed_prompt": detailed_prompt,
            "image_path": generated_image_path,
            # asset_path will be filled in later
        }

    def run(self, prompts_data: Dict[str, List[Dict[str, str]]]) -> None:
        """
        Runs the asset generation pipeline for all prompts in the provided data.
        If the text generator supports prompt packs, generates all prompts at once.
        Then generates all images, offloads the image generator, and generates all assets.
        """
        prompts_output_dir = os.path.join(self.base_output_dir, "prompts")
        all_artifacts = {}

        # 1. Generate all prompts if using a pack generator
        if hasattr(self.text_generator, "generate_pack"):
            logging.info("Using prompt pack generator to generate all prompts at once.")
            pack = self.text_generator.generate_pack(output_dir=prompts_output_dir)
            prompt_list = pack["prompts"]
        else:
            prompt_list = prompts_data['prompts']

        # 2. Generate all images
        for base_prompt_info in prompt_list:
            prompt_name = base_prompt_info['name']
            base_prompt_text = base_prompt_info['text']
            logging.info(f"--- Running pipeline for prompt: {prompt_name} (image generation) ---")

            # If the prompt is a dict (from a pack), use its 'text' directly
            if isinstance(base_prompt_info, dict):
                detailed_prompt = base_prompt_info['text']
            else:
                detailed_prompt = base_prompt_text
            generated_image_path = self.image_generator.generate_image(detailed_prompt, prompt_name=prompt_name)
            logging.info(f"Generated image at: {generated_image_path}")
            all_artifacts[prompt_name] = {
                "prompt_name": prompt_name,
                "base_prompt_text": base_prompt_text,
                "identifier": self.identifier,
                "artifacts": {
                    "detailed_prompt": detailed_prompt,
                    "image_path": generated_image_path
                }
            }

        # 3. Offload image generator to free memory
        logging.info("Offloading image generator to free GPU memory.")
        del self.image_generator
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 4. Generate all assets
        for prompt_name, details_data in all_artifacts.items():
            artifacts = details_data["artifacts"]
            image_path = artifacts["image_path"]
            logging.info(f"--- Running pipeline for prompt: {prompt_name} (asset generation) ---")
            try:
                asset_path = self.asset_generator.generate_asset(image_path, prompt_name)
                artifacts["asset_path"] = asset_path
                logging.info(f"Generated 3D asset at: {asset_path}")
            except Exception as e:
                logging.error(f"Asset generation failed for prompt: {prompt_name}: {e}")
                artifacts["asset_path"] = None

            # Save details after asset generation
            details_path = os.path.join(prompts_output_dir, f"{prompt_name}.json")
            with open(details_path, 'w') as f:
                json.dump(details_data, f, indent=4)
            logging.info(f"Saved prompt details to: {details_path}")
            logging.info(f"--- Pipeline run for {prompt_name} finished ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="3D Asset Generation Pipeline")
    parser.add_argument('--config', type=str, default='configs/pipeline_config.json', help='Path to the pipeline configuration file.')
    parser.add_argument('--secrets', type=str, default='secrets.json', help='Path to the secrets file.')
    parser.add_argument('--prompts', type=str, default='configs/prompts.json', help='Path to the prompts file.')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode using mock generators.')
    parser.add_argument('--identifier', type=str, default=None, help='Optional run identifier for output directory.')
    args = parser.parse_args()

    configure_logging(args.debug)
    config, prompts_data, secrets = get_configs(args)
    if not validate_configs(config, prompts_data, secrets):
        return
    pipeline = Pipeline(config, secrets, args.debug, identifier=args.identifier)
    pipeline.run(prompts_data)


if __name__ == "__main__":
    main()
