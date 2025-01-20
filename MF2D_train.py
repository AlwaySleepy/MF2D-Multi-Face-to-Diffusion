import os
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
import torch
import numpy as np
import cv2
from torch import nn
import argparse
import torchvision
import face_alignment
from PIL import Image
from torch.nn import functional as F
from src import modules
from src import utils
from src.msid import msid_base_patch8_112
from transformers.models.clip.modeling_clip import CLIPTextTransformer,CLIPTextModel
from transformers.models.clip.modeling_clip import _make_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling
from src import mod, modules
import types
import gc
from dataori import get_data_loader, FastComposerDataset
from tqdm import tqdm

def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import (
        Attention,
        AttnProcessor,
        AttnProcessor2_0,
    )

    UNET_LAYER_NAMES = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]

    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        def new_get_attention_scores(module, query, key, attention_mask=None):
            attention_probs = module.old_get_attention_scores(
                query, key, attention_mask
            )
            attention_scores[name] = attention_probs
            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            if isinstance(module.processor, AttnProcessor2_0):
                module.set_processor(AttnProcessor())
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )

    return unet

def fuse_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
):
    object_embeds = object_embeds.to(inputs_embeds.dtype)
    batch_size, max_num_objects = object_embeds.shape[:2]
    seq_length = inputs_embeds.shape[1]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )
    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )

    valid_object_embeds = flat_object_embeds[valid_object_mask]

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)

    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds)
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=1.0, normalize=False):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(
            dim=2
        ) / background_segmaps_sum

        object_loss = (object_token_attn_prob * object_segmaps).sum(
            dim=2
        ) / object_segmaps_sum

        return background_loss - object_loss

def _clear_cross_attention_scores(self):
    if hasattr(self, "cross_attention_scores"):
        keys = list(cross_attention_scores.keys())
        for k in keys:
            del cross_attention_scores[k]

    gc.collect()

def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    object_token_idx,
    object_token_idx_mask,
    loss_fn,
):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    # Resize the object segmentation maps to the size of the cross attention scores
    object_segmaps = F.interpolate(
        object_segmaps, size=(size, size), mode="bilinear", antialias=True
    )  # (b, max_num_objects, size, size)

    object_segmaps = object_segmaps.view(
        b, max_num_objects, -1
    )  # (b, max_num_objects, num_noise_latents)

    num_heads = bxh // b

    cross_attention_scores = cross_attention_scores.view(
        b, num_heads, num_noise_latents, num_text_tokens
    )

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(b, 1, 1, max_num_objects).expand(
            b, num_heads, num_noise_latents, max_num_objects
        ),
    )  # (b, num_heads, num_noise_latents, max_num_objects)

    object_segmaps = (
        object_segmaps.permute(0, 2, 1)
        .unsqueeze(1)
        .expand(b, num_heads, num_noise_latents, max_num_objects)
    )

    loss = loss_fn(object_token_attn_prob, object_segmaps)

    loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
    object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / object_token_cnt).mean()

    return loss

def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    image_token_idx,
    image_token_idx_mask,
    loss_fn,
):
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn
        )
        loss += layer_loss
    return loss / num_layers

#assert torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
unet.requires_grad_(False)
unet.to(torch.float16)
localization_layers = 5
cross_att = True
if cross_att:
    cross_attention_scores = {}
    unet = unet_store_cross_attention_scores(
        unet, cross_attention_scores, localization_layers
    )
    object_localization_loss_fn = BalancedL1Loss()

pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, unet=unet, torch_dtype=torch.float16,safety_checker=None).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

pipe.text_encoder.text_model.forward = mod.forward_texttransformer.__get__(pipe.text_encoder.text_model, CLIPTextTransformer)
pipe.text_encoder.forward = mod.forward_textmodel.__get__(pipe.text_encoder, CLIPTextModel)

img2text = modules.IMG2TEXTwithEXP(384*4,384*4,768)
img2text.load_state_dict(torch.load('checkpoints/mapping.pt',map_location='cpu'))
img2text=img2text.to(device)
img2text.requires_grad_(True)
img2text.train()

msid = msid_base_patch8_112(ext_depthes=[2,5,8,11])
msid.load_state_dict(torch.load('checkpoints/msid.pt'))
msid=msid.to(device)
msid.requires_grad_(False)

optimizer = torch.optim.AdamW(img2text.parameters(), 1e-5)

lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=50,
    )

train_dataset = FastComposerDataset(
    "data/ffhq_wild_files",
    pipe.tokenizer,
    max_num_objects=2,
    num_image_tokens=1,
    object_appear_prob=0.9,
    uncondition_prob=0.1,
    text_only_prob=0,
    object_types="person",
    split="train",
)

train_dataloader = get_data_loader(train_dataset, 16)

# train
loss_record = []
for epoch in range(50):
    train_loss = 0.0
    total = len(train_dataloader)
    denoise_loss = 0.0
    localization_loss = 0.0
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="batch train"):
        num_objects = batch["num_objects"]
        object_pixel_values = batch["object_pixel_values"]
        pos_id = batch["image_token_idx_mask"]
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        image_token_mask = batch["image_token_mask"]

        idvec1 = msid.extract_mlfeat(object_pixel_values[:,0,:,:,:].to(device).float(),[2,5,8,11])
        tokenized_identity_first1, tokenized_identity_last1 = img2text(idvec1,exp=None)

        idvec2 = msid.extract_mlfeat(object_pixel_values[:,1,:,:,:].to(device).float(),[2,5,8,11])
        tokenized_identity_first2, tokenized_identity_last2 = img2text(idvec2,exp=None)
        
        tokenized_identity_first = torch.cat((tokenized_identity_first1.unsqueeze(1), tokenized_identity_first2.unsqueeze(1)),dim=1)
        encoder_hidden_states = mod.forward_texttransformer(pipe.text_encoder.text_model, input_ids=input_ids.to(device))[0]
        
        encoder_hidden_states = fuse_object_embeddings(
            encoder_hidden_states.to(device), image_token_mask.to(device), tokenized_identity_first, num_objects.to(device)
        )

        vae_input = pixel_values.to(torch.float16)
        #print("vae", vae_input.shape)

        latents = pipe.vae.encode(vae_input).latent_dist.sample()
        #print(latents.shape)
        latents = latents * pipe.vae.config.scaling_factor
        #print("latent", latents.shape)

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
                    0, pipe.scheduler.num_train_timesteps, (bsz,), device=latents.device
                )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

        pred = unet(noisy_latents.to(dtype=torch.float16), timesteps, encoder_hidden_states.to(dtype=torch.float16))["sample"]

        # Get the target for loss depending on the prediction type
        if pipe.scheduler.config.prediction_type == "epsilon":
            target = noise
        elif pipe.scheduler.config.prediction_type == "v_prediction":
            target = pipe.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {pipe.scheduler.config.prediction_type}"
            )

        if torch.rand(1) < 0.5:
            object_segmaps = batch["object_segmaps"]
            mask = (object_segmaps.sum(dim=1) > 0).float()
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(pred.shape[-2], pred.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred = pred * mask
            target = target * mask
        #print(pred)
        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        return_dict = {"denoise_loss": denoise_loss}

        if cross_att:
            object_segmaps = batch["object_segmaps"]
            image_token_idx = batch["image_token_idx"]
            image_token_idx_mask = batch["image_token_idx_mask"]
            localization_loss = get_object_localization_loss(
                cross_attention_scores,
                object_segmaps.to(device),
                image_token_idx.to(device),
                image_token_idx_mask.to(device),
                object_localization_loss_fn,
            )
            return_dict["localization_loss"] = localization_loss
            loss = 0.01 * localization_loss + denoise_loss
            _clear_cross_attention_scores(cross_attention_scores)
        else:
            loss = denoise_loss

        # Backpropagate
        loss.backward()
        optimizer.step()
        #lr_scheduler.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    loss_record.append(train_loss/total)
    print(f"epoch:{epoch}  loss:{train_loss/total}") 
    PATH = f'./fmap/fmap_epoch{epoch}.pth'
    torch.save(img2text.state_dict(), PATH)
    with open("./training_loss0.txt", 'a') as train_los:
        train_los.write(f"{train_loss/total}\n")
with open("./training_loss0.txt", 'a') as train_los:
    train_los.write(str(loss_record))
