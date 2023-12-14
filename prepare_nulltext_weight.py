from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import *
import torch
import torch.nn.functional as F
from nulltext_attention import NullTextAttention


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")

for name, module in pipe.unet.named_modules():
    if name.endswith("attn2"):
        _ = NullTextAttention(module)

torch.save(pipe.unet.state_dict(), "stable-diffusion-v1-5_null-text.pt")
