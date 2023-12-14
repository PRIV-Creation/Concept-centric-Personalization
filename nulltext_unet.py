from diffusers import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from nulltext_attention import NullTextAttention
import torch


class UNet2DConditionModel_Nulltext(UNet2DConditionModel):
    @classmethod
    def from_pretrained(cls, null_text_weight, *args, **kwargs):
        unet = super().from_pretrained(*args, **kwargs)
        for name, module in unet.named_modules():
            if name.endswith("attn2"):
                _ = NullTextAttention(module, initial=True)
                channel = module.to_q.in_features
                del module.to_q
                del module.to_k
                del module.to_v
                del module.to_out
                module.register_buffer("null_text_feature", torch.zeros([1, 1, channel]))
            if isinstance(module, BasicTransformerBlock):
                module.norm2 = torch.nn.Identity()
        unet.load_state_dict(null_text_weight, strict=True)
        return unet


if __name__ == '__main__':
    unet = UNet2DConditionModel_Nulltext.from_pretrained(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        subfolder='unet',
        null_text_weight="stable-diffusion-v1-5_null-text.pt"
    )