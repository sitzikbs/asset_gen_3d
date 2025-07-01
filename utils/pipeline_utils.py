import logging
import uuid
from typing import Dict, Any, Optional, Tuple
from utils.file_utils import load_json_config

def get_identifier(debug: bool, identifier: Optional[str]) -> str:
    """
    Returns the experiment identifier. If not provided, returns 'debug_run' in debug mode or a new UUID otherwise.
    """
    if identifier:
        return identifier
    if debug:
        return "debug_run"
    logging.info("No experiment identifier provided, generating a new unique identifier for this run.")
    return str(uuid.uuid4())


def configure_logging(debug: bool) -> None:
    """
    Configures the logging level and format based on debug mode.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


def get_configs(args) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Loads configuration, prompts, and secrets based on the provided arguments and debug mode.
    Returns a tuple: (config, prompts_data, secrets)
    """
    if args.debug:
        config = {}
        prompts_data = {"prompts": [{"name": "debug_prompt", "text": "A test prompt for debugging."}]}
        secrets = {}
    else:
        config = load_json_config(args.config)
        prompts_data = load_json_config(args.prompts)
        secrets = load_json_config(args.secrets)
    return config, prompts_data, secrets


def validate_configs(config, prompts_data, secrets) -> bool:
    """
    Validates that all required configs are loaded and prompts are present.
    Returns True if valid, False otherwise.
    """
    if (config is None or prompts_data is None or secrets is None or not prompts_data.get('prompts')):
        logging.error("Exiting due to configuration or secrets errors.")
        return False
    return True
