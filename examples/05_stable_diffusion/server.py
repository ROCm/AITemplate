
from fastapi import FastAPI
from typing import List,Union
from pydantic import BaseModel
from src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline
import torch
from tqdm import tqdm
from PIL import Image
import io
from fastapi import Response
from diffusers import EulerDiscreteScheduler

class Item(BaseModel):
    prompt: Union[str, List[str]]
    img_height: int = 512
    img_width: int = 512
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

app = FastAPI()

local_dir ="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2/"
pipe = StableDiffusionAITPipeline.from_pretrained(
        local_dir,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            local_dir, subfolder="scheduler"
        ),
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

torch_device = torch.device("cuda:0")

@app.post("/predict/")
async def predict(input_api: Item):
    with torch.autocast("cuda"):
        image = pipe(
            input_api.prompt,
            height = input_api.img_height,
            width = input_api.img_width,
            num_inference_steps = input_api.num_inference_steps,
            guidance_scale = input_api.guidance_scale).images[0]
    
    
    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        image.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im_bytes, headers=headers, media_type='image/png')
