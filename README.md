## In-domain Generation with Diffusion Models
The official code of "Image is All You Need to Empower Large-scale Diffusion Models for In-Domain Generation".

![img.pdf](assets/main.png)

### 1. Code for Null-text UNet

#### 1.1 Prepare null-text checkpoint

```bash
python prepare_nulltext_checkpoint.py
```

#### 1.2 Construct null-text UNet

```bash
python nulltext_unet.py
```

### 2. In-domain Generation with Multi-Guidance

See `generation_with_nulltext_model.py` for details.

The core code of GCFG are as follows:

```python
noise_pred_text = self.unet(
                    latent_model_input[1:],
                    t,
                    encoder_hidden_states=prompt_embeds[2:3],
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

noise_pred_text_ori = self.unet1(
    latent_model_input[1:],
    t,
    encoder_hidden_states=prompt_embeds[3:4],
    cross_attention_kwargs=cross_attention_kwargs,
    return_dict=False,
)[0]

noise_pred_uncond = self.unet0(
    latent_model_input[:1],
    t,
    encoder_hidden_states=prompt_embeds[:1],
    cross_attention_kwargs=cross_attention_kwargs,
    return_dict=False,
)[0]

# perform guidance
if do_classifier_free_guidance:
    noise_pred = noise_pred_uncond + \
                 guidance_scale * (noise_pred_text - noise_pred_uncond) + \
                 guidance_scale_ori * (noise_pred_text_ori - noise_pred_uncond)
```

where ```self.unet0``` is SD1.5 for unconditional guidance, ```self.unet``` is in-domain diffusion model for domain guidance, and ```self.unet1``` is SD1.5 or customized SD for control guidance.

### TODO
- [ ] Updating training codes in [UniDiffusion](https://github.com/PRIV-Creation/UniDiffusion).
- [ ] Results on SDXL and SD3.
