import os
from typing import Any, Dict, Optional, Literal
from modules.image_generators.base_image_generator import BaseImageGenerator
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from transformers import BitsAndBytesConfig, T5EncoderModel, CLIPTextModel


class FluxImageGenerator(BaseImageGenerator):
    """
    Image generator using any FLUX.1 model via Hugging Face diffusers.
    Implements the BaseImageGenerator interface.
    """
    def __init__(self, secrets: Optional[Dict[str, Any]] = None, output_dir: str = "output/images", model_id: str = "black-forest-labs/FLUX.1-schnell", quantization_type: Literal["none", "4bit", "8bit"] = "none"):
        super().__init__(secrets or {}, output_dir)
        self.model_id = model_id
        self.quantization_type = quantization_type
        # Detect device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.bfloat32


        if not self.quantization_type == "none":
            if self.quantization_type == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    # bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                )
            elif self.quantization_type == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    # bnb_8bit_use_double_quant=True,
                    bnb_8bit_compute_dtype=torch_dtype,
                )
            else:
                raise ValueError(f"Unsupported quantization type: {self.quantization_type}. Use 'none', '4bit', or '8bit'.")
            
            transformer = FluxTransformer2DModel.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                subfolder="transformer",
            )
            text_encoder_two = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )
            text_encoder_one = CLIPTextModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )
            vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder="vae",
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
            )

            self.pipe = FluxPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                vae=vae,
                torch_dtype=torch_dtype,
            )

            self.pipe.to(self.device)
           
        else: 
            self.pipe = FluxPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            )
            self.pipe.enable_sequential_cpu_offload()
        



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

        with torch.no_grad():
            image = self.pipe(**pipe_args).images[0]
        image.save(image_path)
        return image_path


# Test function for FluxImageGenerator

def test_flux_image_generator(show_image: bool = True, cleanup: bool = False, model_id: str = "black-forest-labs/FLUX.1-schnell", quantization_type: Literal["none", "4bit", "8bit"] = "none"):
    """Test the FluxImageGenerator with the specified parameters.

    Args:
        show_image (bool, optional): Whether to display the generated image. Defaults to True.
        cleanup (bool, optional): Whether to clean up the generated image files. Defaults to False.
        model_id (str, optional): The model ID to use for image generation. Defaults to "black-forest-labs/FLUX.1-schnell".
    """

    print(f"Testing FluxImageGenerator with model: {model_id}")
    
    debug_dir = "outputs/debug/image_gen"
    os.makedirs(debug_dir, exist_ok=True)

    generator = FluxImageGenerator(secrets={}, output_dir=debug_dir, model_id=model_id, quantization_type=quantization_type)

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

    test_flux_image_generator(show_image=True, cleanup=False, model_id="black-forest-labs/FLUX.1-schnell", quantization_type="8bit")
    # test_flux_image_generator(show_image=True, cleanup=False, model_id="black-forest-labs/FLUX.1-dev")