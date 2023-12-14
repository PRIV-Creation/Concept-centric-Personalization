## Concept-centric Personalization
The official code of "Concept-centric Personalization with Large-scale Diffusion Priors".

The training code and models will be released later. We here provide the code for null-text UNet and GCFG.

### 1. Code for Null-text UNet

#### 1.1 Prepare null-text checkpoint

```bash
python prepare_nulltext_checkpoint.py
```

#### 1.2 Construct null-text UNet

```bash
python nulltext_unet.py
```

### 2. GCFG

#### 2.1 Concept-centric Generation with GCFG

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

where ```self.unet0``` is SD1.5 for unconditional guidance, ```self.unet``` is concept-centric diffusion model for concept guidance, and ```self.unet1``` is SD1.5 or customized SD for control guidance.

#### 2.2 Generic Generation with GCFG

See `gcfg.py` for details.

The core code are as follows:

```bash
if do_classifier_free_guidance:
    noise_pred = (1 - sum(weight)) * noise_pred_uncond
    for w, p in zip(weight, noise_pred_text):
        noise_pred += w * p[None]
```

### Citation
```bibtex
@misc{cao2023conceptcentric,
      title={Concept-centric Personalization with Large-scale Diffusion Priors}, 
      author={Pu Cao and Lu Yang and Feng Zhou and Tianrui Huang and Qing Song},
      year={2023},
      eprint={2312.08195},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```