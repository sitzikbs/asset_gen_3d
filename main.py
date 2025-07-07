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
        self.generation_mode = config.get("generation_mode", "pack")
        self.num_packs = int(config.get("num_packs", 1))
        # Set self.run to the correct method for main()
        if self.generation_mode == "single_asset":
            self.run = self.run_single_asset
        elif self.generation_mode == "pack":
            self.run = lambda *_: self.run_multi_pack(num_packs=self.num_packs)
        else:
            raise ValueError(f"Unknown generation_mode: {self.generation_mode}")
        self.pack_size = int(config.get("pack_size", 3))
        if debug:
            logging.info("Running in debug mode. Using mock generators.")
            from modules.text_generators.mock_text_generator import MockTextGenerator
            from modules.image_generators.mock_image_generator import MockImageGenerator
            from modules.asset_generators.mock_asset_generator import MockAssetGenerator
            self.text_generator = MockTextGenerator(secrets={}, output_dir=os.path.join(self.base_output_dir, "prompts"))
            self.image_generator = MockImageGenerator(secrets={}, output_dir=os.path.join(self.base_output_dir, "images"))
            self.asset_generator = MockAssetGenerator(secrets={}, output_dir=os.path.join(self.base_output_dir, "assets"))
        else:
            text_gen_cfg = dict(config.get('text_generator', {}))
            self.text_generator = self._load_generator('text_generators', text_gen_cfg, secrets, os.path.join(self.base_output_dir, "prompts"))
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

    def run_single_asset(self):
        """Generate a single prompt, image, and asset in a flat output directory."""
        prompts_output_dir = os.path.join(self.base_output_dir, "prompts")
        images_output_dir = os.path.join(self.base_output_dir, "images")
        assets_output_dir = os.path.join(self.base_output_dir, "assets")
        os.makedirs(prompts_output_dir, exist_ok=True)
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(assets_output_dir, exist_ok=True)

        # Generate a single prompt
        if hasattr(self.text_generator, "generate_single_prompt"):
            # Use the modular single prompt method if available
            prompt_dict = self.text_generator.generate_single_prompt(
                object_type=None, genre=None, style=None, material=None, color_palette=None, idx=0
            )
        else:
            # Fallback: use generate_prompt and take the first
            prompt_dict = self.text_generator.generate_prompt()[0]

        prompt_name = prompt_dict["name"]
        prompt_text = prompt_dict["text"]
        
        # Save prompt
        with open(os.path.join(prompts_output_dir, f"{prompt_name}.json"), "w") as f:
            json.dump(prompt_dict, f, indent=2)
        # Generate image
        image_path = self.image_generator.generate_image(prompt_text, prompt_name=prompt_name)
        # Generate asset
        asset_path = self.asset_generator.generate_asset(image_path, prompt_name)
        # Save summary
        summary = {
            "prompt": prompt_dict,
            "image_path": image_path,
            "asset_path": asset_path
        }
        
        with open(os.path.join(self.base_output_dir, f"{prompt_name}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"Single asset generation complete: {summary}")

    def run_multi_pack(self, num_packs: int = 1):
        """Generate multiple packs, then organize all outputs into the correct pack directories."""
        import shutil
        experiment_dir = self.base_output_dir
        final_packs_dir = os.path.join(experiment_dir, "final_packs")
        os.makedirs(final_packs_dir, exist_ok=True)
        self.pack_outputs = {}
        all_prompts = []
        main_prompts_dir = os.path.join(self.base_output_dir, "prompts")
        os.makedirs(main_prompts_dir, exist_ok=True)
        for i in range(num_packs):
            pack_name = f"pack_{i+1}"
            pack_dir = os.path.join(final_packs_dir, pack_name)
            os.makedirs(pack_dir, exist_ok=True)
            # Generate prompts for this pack, passing pack_size from pipeline config (do not save prompts.json automatically)
            pack = self.text_generator.generate_pack(pack_size=self.pack_size, output_dir=None)
            prompt_list = pack["prompts"]
            all_prompts.extend(prompt_list)
            # Save the prompts for this pack in the pack dir as pack_X_prompts.json
            pack_prompts_json_path = os.path.join(pack_dir, f"{pack_name}_prompts.json")
            with open(pack_prompts_json_path, "w") as f:
                json.dump({"prompts": prompt_list}, f, indent=2)
            # Save pack metadata (theme info) in the pack dir
            pack_metadata_path = os.path.join(pack_dir, "pack_metadata.json")
            with open(pack_metadata_path, "w") as f:
                json.dump(pack["theme"], f, indent=2)
            all_artifacts = {}
            asset_files = []
            prompt_files = []

            # Move image generator to GPU
            self.image_generator.to_gpu()

            # Generate images (images are not saved in pack dir, but can be if needed)
            for base_prompt_info in prompt_list:
                prompt_name = base_prompt_info['name']
                prompt_text = base_prompt_info['text']
                logging.info(f"[Pack {i+1}] Generating image for: {prompt_name}")
                image_path = self.image_generator.generate_image(prompt_text, prompt_name=prompt_name)
                all_artifacts[prompt_name] = {
                    "prompt_name": prompt_name,
                    "base_prompt_text": prompt_text,
                    "identifier": self.identifier,
                    "artifacts": {
                        "detailed_prompt": prompt_text,
                        "image_path": image_path
                    }
                }
            # Move image generator to CPU to free GPU memory and clear caches
            self.image_generator.to_cpu()

            # Group all memory management here for clarity
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Generate assets and save prompt+asset details in pack dir
            for prompt_name, details_data in all_artifacts.items():
                artifacts = details_data["artifacts"]
                image_path = artifacts["image_path"]
                logging.info(f"[Pack {i+1}] Generating asset for: {prompt_name}")
                try:
                    asset_path = self.asset_generator.generate_asset(image_path, prompt_name)
                    artifacts["asset_path"] = asset_path
                    if asset_path and os.path.isfile(asset_path):
                        asset_files.append(asset_path)
                except Exception as e:
                    logging.error(f"Asset generation failed for prompt: {prompt_name}: {e}")
                    artifacts["asset_path"] = None
                # Save details directly in the pack dir
                details_path = os.path.join(pack_dir, f"{prompt_name}.json")
                with open(details_path, 'w') as f:
                    json.dump(details_data, f, indent=4)
                prompt_files.append(details_path)
            # Track all outputs for this pack
            self.pack_outputs[pack_name] = {
                "pack_dir": pack_dir,
                "asset_files": asset_files,
                "prompt_files": prompt_files,
                "prompts_json": pack_prompts_json_path,
                "pack_metadata": pack_metadata_path
            }
            logging.info(f"Pack {i+1} generation complete: {pack_dir}")
        # Save all prompts for all packs in the main prompts dir as prompts.json
        main_prompts_json_path = os.path.join(main_prompts_dir, "prompts.json")
        with open(main_prompts_json_path, "w") as f:
            json.dump({"prompts": all_prompts}, f, indent=2)
        # Organize all outputs (copy assets to pack dir, etc.)
        self.organize_pack_outputs()

    def organize_pack_outputs(self):
        """Copy/move all relevant files (assets, prompts, metadata) into their pack directories."""
        import shutil
        for pack_name, outputs in self.pack_outputs.items():
            pack_dir = outputs["pack_dir"]
            # Copy asset files to pack dir (if not already there)
            for asset_path in outputs["asset_files"]:
                if asset_path and os.path.isfile(asset_path):
                    dest_path = os.path.join(pack_dir, os.path.basename(asset_path))
                    if os.path.abspath(asset_path) != os.path.abspath(dest_path):
                        shutil.copy2(asset_path, dest_path)
            # Prompts and metadata are already saved in the correct place
        logging.info("All pack outputs organized.")


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
