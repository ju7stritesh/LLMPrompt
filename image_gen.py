from diffusers import StableDiffusionXLPipeline
import torch
import cv2
import numpy as np
from PIL import Image

pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()
prompt = "A cute cat" # Your prompt here
neg_prompt = "ugly, blurry, poor quality" # Negative prompt here
image = pipe(prompt=prompt, negative_prompt=neg_prompt, width = 720, height = 720, num_inference_steps = 15).images[0]
image.show()
image.save("cat.jpg")

