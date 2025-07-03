import os
from typing import Any, Dict, Optional
from modules.image_generators.base_image_generator import BaseImageGenerator
import torch
from diffusers import FluxPipeline

def get_model_size_in_ram(model):
    """calculate the real size of a pyutorch model in RAM for debugging

    """
    total_size_bytes = 0
    for param in model.parameters():
        total_size_bytes += param.nelement() * param.element_size()
    for buffer in model.buffers():
        total_size_bytes += buffer.nelement() * buffer.element_size()
    # Convert bytes to megabytes
    total_size_mbytes = total_size_bytes / (1024 * 1024)  # Convert to MB
    return total_size_mbytes


class FluxImageGenerator(BaseImageGenerator):
    """
    Image generator using any FLUX.1 model via Hugging Face diffusers.
    Implements the BaseImageGenerator interface.
    """
    def __init__(self, secrets: Optional[Dict[str, Any]] = None, output_dir: str = "output/images", model_id: str = "black-forest-labs/FLUX.1-schnell"):
        super().__init__(secrets or {}, output_dir)
        self.model_id = model_id
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.bfloat32
        self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            load_in_4bit=True if self.device.type == "cuda" else False,
        )
        
        transformer_size_mb = get_model_size_in_ram(self.pipe.transformer)
        vae_size_mb = get_model_size_in_ram(self.pipe.vae)
        total_size_mb = transformer_size_mb + vae_size_mb
        print(f"Model sizes - Transformer: {transformer_size_mb:.2f} MB, VAE: {vae_size_mb:.2f} MB, Total: {total_size_mb:.2f} MB")
        breakpoint()
        self.pipe.to(self.device)
        self.pipe.enable_model_cpu_offload()
        # self.pipe = torch.compile(self.pipe, mode="reduce-overhead")


    def generate_image(self, prompt: str, prompt_name: Optional[str] = None, **kwargs) -> str:
        """
        Generates an image using the selected FLUX.1 model.
        Accepts additional kwargs for model-specific parameters.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        image_filename = f"{prompt_name or 'flux_image'}.png"
        image_path = os.path.join(self.output_dir, image_filename)

        # Allow for model-specific parameters
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        pipe_args = dict(
            prompt=prompt,
            guidance_scale=kwargs.get('guidance_scale', 0.0),
            num_inference_steps=kwargs.get('num_inference_steps', 4),
            max_sequence_length=kwargs.get('max_sequence_length', 256),
            generator=torch.Generator(generator_device).manual_seed(0)
        )
        
        # For dev model, allow height/width and different defaults
        if self.model_id.endswith('dev'):
            pipe_args['height'] = kwargs.get('height', 1024)
            pipe_args['width'] = kwargs.get('width', 1024)
            pipe_args['guidance_scale'] = kwargs.get('guidance_scale', 3.5)
            pipe_args['num_inference_steps'] = kwargs.get('num_inference_steps', 50)
            pipe_args['max_sequence_length'] = kwargs.get('max_sequence_length', 512)
        
        print(f"Generating image with parameters: {pipe_args}")
        image = self.pipe(**pipe_args).images[0]
        image.save(image_path)
        return image_path


# Test function for FluxImageGenerator

def test_flux_image_generator(show_image: bool = True, cleanup: bool = False, model_id: str = "black-forest-labs/FLUX.1-schnell"):
    """Test the FluxImageGenerator with the specified parameters.

    Args:
        show_image (bool, optional): Whether to display the generated image. Defaults to True.
        cleanup (bool, optional): Whether to clean up the generated image files. Defaults to False.
        model_id (str, optional): The model ID to use for image generation. Defaults to "black-forest-labs/FLUX.1-schnell".
    """

    print(f"Testing FluxImageGenerator with model: {model_id}")
    
    debug_dir = "outputs/debug/image_gen"
    os.makedirs(debug_dir, exist_ok=True)

    generator = FluxImageGenerator(secrets={}, output_dir=debug_dir, model_id=model_id)
    prompt = "A medieval treasure chest in a computer graphics style with intricate details, " \
              "glowing runes, and a mystical aura, set against a white background."
    prompt_name = f"test_medieval_treasure_chest_{model_id.split('/')[-1]}"
    image_path = generator.generate_image(prompt, prompt_name)
    assert os.path.exists(image_path), f"Image file was not created: {image_path}"
    print(f"Generated image for prompt '{prompt_name}' at: {image_path}")

    if show_image:
        try:
            from PIL import Image
            img = Image.open(image_path)
            img.show()
        except ImportError:
            print("PIL not installed, cannot display image.")

    if cleanup:
        os.remove(image_path)
        os.rmdir(os.path.dirname(image_path))
        print("Image and directory cleaned up.")
    else:
        print(f"Image kept at: {image_path}")


if __name__ == "__main__":
    # Example: test with schnell and dev
    test_flux_image_generator(show_image=True, cleanup=False, model_id="black-forest-labs/FLUX.1-schnell")
    # test_flux_image_generator(show_image=True, cleanup=False, model_id="black-forest-labs/FLUX.1-dev")
