import json
import logging
from typing import Dict, Any, Optional

def load_json_config(file_path: str) -> Optional[Dict[str, Any]]:
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
