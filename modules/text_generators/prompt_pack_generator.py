import json
import random
from typing import List, Dict, Any, Optional

class PromptPackGenerator:

    """Generates themed prompt packs for 3D asset generation.
    Uses a configuration file to define themes, styles, materials, and object types.
    Prompts are structured for compatibility with the asset generation pipeline.
    """

    def __init__(self, config_path: str, seed: Optional[int] = None, pack_size: int = 3, secrets: Optional[dict] = None, **kwargs):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.rng = random.Random(seed)
        self.seed = seed
        self.pack_size = int(pack_size)
        self.secrets = secrets
        self.prompt_suffix = self.config.get("prompt_suffix", "")

    def save_prompts(self, pack: Dict[str, Any], output_dir: str) -> str:
        """
        Save prompts in pipeline-compatible format to a file in output_dir.
        Returns the path to the saved file.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        # Structure prompts for pipeline: name, text, metadata
        prompts_data = {
            "prompts": [
                {
                    "name": p["name"],
                    "text": p["text"],
                    "metadata": p["metadata"]
                }
                for p in pack["prompts"]
            ]
        }
        out_path = os.path.join(output_dir, "prompts.json")
        with open(out_path, "w") as f:
            json.dump(prompts_data, f, indent=2)
        return out_path
    
    def generate_prompt(self, *args, pack_size: Optional[int] = None, output_dir: Optional[str] = None, **kwargs):
        """
        Generates a prompt pack and returns the list of prompts.
        Ignores positional arguments (for pipeline compatibility), uses only keyword arguments for pack_size/output_dir.
        """
        pack = self.generate_pack(pack_size=pack_size, output_dir=output_dir, **kwargs)
        return pack["prompts"]
    
    def generate_pack(self, pack_size: Optional[int] = None, output_dir: Optional[str] = None, **kwargs) -> Dict[str, Any]:

        if pack_size is None:
            pack_size = self.pack_size
        # Ensure pack_size is always an int
        try:
            pack_size = int(pack_size)
        except Exception:
            raise ValueError(f"pack_size must be an integer, got {pack_size} of type {type(pack_size)}")
        
        genres = self.config["genres"]
        styles = self.config["styles"]
        materials = self.config["materials"]
        color_palettes = self.config["color_palettes"]
        object_types_by_genre = self.config["object_types"]
        universal_object_types = self.config["universal_object_types"]

        genre = self.rng.choice(genres)
        style = self.rng.choice(styles)
        material = self.rng.choice(materials)
        color_palette = self.rng.choice(color_palettes)

        # Select object types: majority from genre, some from universal pool
        if genre in object_types_by_genre:
            genre_objects = list(object_types_by_genre[genre])
        else:
            genre_objects = []
        self.rng.shuffle(genre_objects)
        # If no genre-specific objects, use all from universal
        if not genre_objects:
            object_types = list(universal_object_types)
            self.rng.shuffle(object_types)
            object_types = object_types[:pack_size]
        else:
            num_genre = max(pack_size - 2, 1)
            num_universal = pack_size - num_genre
            selected_genre_objs = genre_objects[:num_genre]
            universal_candidates = [o for o in universal_object_types if o not in selected_genre_objs]
            self.rng.shuffle(universal_candidates)
            selected_universal_objs = universal_candidates[:num_universal]
            object_types = selected_genre_objs + selected_universal_objs
            self.rng.shuffle(object_types)

        prompts = []
        for i, obj in enumerate(object_types):
            prompt_text = self.compose_prompt(obj, genre, style, material, color_palette)
            prompt_dict = {
                "name": f"{genre}_{style}_{i+1}_{obj}",
                "text": prompt_text,
                "metadata": {
                    "object_type": obj,
                    "genre": genre,
                    "style": style,
                    "material": material,
                    "color_palette": color_palette,
                    "prompt": prompt_text,
                    "seed": self.seed
                }
            }
            prompts.append(prompt_dict)

        pack = {
            "theme": {
                "genre": genre,
                "style": style,
                "material": material,
                "color_palette": color_palette,
                "seed": self.seed
            },
            "prompts": prompts
        }
        if output_dir is not None:
            self.save_prompts(pack, output_dir)
        return pack

    def compose_prompt(self, object_type: str, genre: str, style: str, material: str, color_palette: str) -> str:
        base = (
            f"A {style} {object_type} for a {genre} game, made of {material}, "
            f"with a {color_palette} color palette. Highly detailed, game-ready."
        )
        if self.prompt_suffix:
            return f"{base} {self.prompt_suffix}"
        return base

if __name__ == "__main__":
    # Example usage and test
    generator = PromptPackGenerator(
        config_path="configs/asset_pack_categories.json",
        seed=42
    )
    pack = generator.generate_pack(pack_size=8, output_dir="outputs/prompts")
    print("Generated Asset Pack Theme:")
    print(json.dumps(pack["theme"], indent=2))
    out_path = "outputs/prompts/prompts.json"
    print(f"\nPrompts saved to: {out_path}")

    # Show a preview
    with open(out_path) as f:
        prompts_data = json.load(f)
    print("\nPrompt file preview:")
    for i, p in enumerate(prompts_data["prompts"], 1):
        print(f"{i}. {p['name']}: {p['text']}")
