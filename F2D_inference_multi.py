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
from dataori import get_data_loader, FastComposerDataset, DemoDataset
from tqdm import tqdm

with torch.no_grad():
    assert torch.cuda.is_available() # you must use GPU e.g. T4
    device = 'cuda'


    pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16,safety_checker=None).to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.text_encoder.text_model.forward = mod.forward_texttransformer.__get__(pipe.text_encoder.text_model, CLIPTextTransformer)
    pipe.text_encoder.forward = mod.forward_textmodel.__get__(pipe.text_encoder, CLIPTextModel)

    img2text = modules.IMG2TEXTwithEXP(384*4,384*4,768)
    #img2text.load_state_dict(torch.load('./fmap/fmap_epoch49.pth',map_location='cpu'))
    img2text.load_state_dict(torch.load('checkpoints/mapping.pt',map_location='cpu'))
    img2text=img2text.to(device)
    img2text.eval()

    msid = msid_base_patch8_112(ext_depthes=[2,5,8,11])
    msid.load_state_dict(torch.load('checkpoints/msid.pt'))
    msid=msid.to(device)
    msid.eval()

    input_prompt = 'f l and g l are happy couple on the beach' # You must input "f l" to represent the subject S*
    guidance_scale = 10.0 # classifier-free guidance
    n_samples = 8 # num of images to generate
    img_path1 = 'data/input/0.jpg' # path to input image
    img_path2 = 'data/input/3.jpg' 
    output_dir = "data/f2d"

    identifier1='f'
    ids = pipe.tokenizer(
                    input_prompt,
                    padding="do_not_pad",
                    truncation=True,
                    max_length=pipe.tokenizer.model_max_length,
                ).input_ids
    placeholder_token_id1=pipe.tokenizer(
                    identifier1,
                    padding="do_not_pad",
                    truncation=True,
                    max_length=pipe.tokenizer.model_max_length,
                ).input_ids[1]
    assert placeholder_token_id1 in ids,'identifier1 does not exist in prompt'
    pos_id1 = ids.index(placeholder_token_id1)

    identifier2='g'
    placeholder_token_id2=pipe.tokenizer(
                    identifier2,
                    padding="do_not_pad",
                    truncation=True,
                    max_length=pipe.tokenizer.model_max_length,
                ).input_ids[1]
    assert placeholder_token_id2 in ids,'identifier2 does not exist in prompt'
    pos_id2 = ids.index(placeholder_token_id2)

    input_ids = pipe.tokenizer.pad(
            {"input_ids": [ids]},
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
    

    detector=face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,flip_input=False,device='cuda' if torch.cuda.is_available() else 'cpu')
    lmk1=np.array(detector.get_landmarks(img_path1))[0]
    lmk2=np.array(detector.get_landmarks(img_path2))[0]
    img1 = np.array(Image.open(img_path1).convert('RGB'))
    img2 = np.array(Image.open(img_path2).convert('RGB'))
    with torch.no_grad():
        M1=utils.align(lmk1)
        img1=utils.warp_img(img1,M1,(112,112))/255
        img1=torch.tensor(img1).permute(2,0,1).unsqueeze(0)
        img1=(img1-0.5)/0.5
        idvec1 = msid.extract_mlfeat(img1.to(device).float(),[2,5,8,11])
        tokenized_identity_first1, tokenized_identity_last1 = img2text(idvec1,exp=None)
        hidden_states = utils.get_clip_hidden_states(input_ids.to(device),pipe.text_encoder).to(dtype=torch.float32)
        hidden_states[[0], [pos_id1]]=tokenized_identity_first1.to(dtype=torch.float32)
        hidden_states[[0], [pos_id1+1]]=tokenized_identity_last1.to(dtype=torch.float32)
        #pos_eot = input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1)

        M2=utils.align(lmk2)
        img2=utils.warp_img(img2,M2,(112,112))/255
        img2=torch.tensor(img2).permute(2,0,1).unsqueeze(0)
        img2=(img2-0.5)/0.5
        idvec2 = msid.extract_mlfeat(img2.to(device).float(),[2,5,8,11])
        tokenized_identity_first2, tokenized_identity_last2 = img2text(idvec2,exp=None)
        #hidden_states = utils.get_clip_hidden_states(input_ids.to(device),pipe.text_encoder).to(dtype=torch.float32)
        hidden_states[[0], [pos_id2]]=tokenized_identity_first2.to(dtype=torch.float32)
        hidden_states[[0], [pos_id2+1]]=tokenized_identity_last2.to(dtype=torch.float32)
        pos_eot = input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1)
    with torch.autocast("cuda"):
        with torch.no_grad():
            encoder_hidden_states = pipe.text_encoder(hidden_states=hidden_states, pos_eot=pos_eot)[0]
    generator = torch.Generator(device).manual_seed(0)
    image = pipe(prompt_embeds=encoder_hidden_states, num_inference_steps=30, guidance_scale=guidance_scale,generator=generator,num_images_per_prompt=n_samples).images#[0]
    for instance_id in range(8):
        image[instance_id].save(
            os.path.join(
                output_dir,
                f"output_{instance_id}.png",
            )
        )
