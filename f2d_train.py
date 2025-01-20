from data import FastComposerDataset, get_data_loader
from model import BiSeNet
from src import modules
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from src import utils
import face_alignment
from src.msid import msid_base_patch8_112
from torch.nn import functional as F

from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler, UNet2DConditionModel
from src import mod
from transformers.models.clip.modeling_clip import CLIPTextTransformer,CLIPTextModel
from diffusers.optimization import get_scheduler
from transformers.modeling_outputs import BaseModelOutputWithPooling



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
    # print(num_objects.device)
    # print(flat_object_embeds.device)
    valid_object_mask = (

        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects.to("cuda:0")[:, None]
    )
    # print(valid_object_mask.shape)
    flat_embeds=flat_object_embeds.flatten()
    valid_object_embeds = flat_embeds[valid_object_mask.flatten()]

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    # print(image_token_embeds.shape)
    # print(valid_object_embeds.shape)
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds.T)

    inputs_embeds.masked_scatter_(image_token_mask.to("cuda:0")[:, None], valid_object_embeds)
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float32,safety_checker=None).to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet").to("cuda:0")
unet.requires_grad_(True)
# unet.to(torch.float16)


# dataset: FFHQ
dataset = FastComposerDataset(
    "data/ffhq_wild_files",
    pipe.tokenizer,
    max_num_objects=1,
    num_image_tokens=1,
    object_appear_prob=0.9,
    uncondition_prob=0.1,
    text_only_prob=0,
    object_types="person",
    split="train",
    cgdr=True,
)
# data augmentation: horizontal flip
transforms = []
dataloader =get_data_loader(dataset, 8) 

# params
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
lr = 10e-5
iteration = 100000

msid = msid_base_patch8_112(ext_depthes=[2,5,8,11])
msid.load_state_dict(torch.load('checkpoints/msid.pt'))

msid=msid.to(device)
msid.eval()


#build f2d pipeline
#__get__：对于实例pipe.text_encoder，类型是 CLIPTextModel
pipe.text_encoder.text_model.forward = mod.forward_texttransformer.__get__(pipe.text_encoder.text_model, CLIPTextTransformer)
pipe.text_encoder.forward = mod.forward_textmodel.__get__(pipe.text_encoder, CLIPTextModel)

# 3D face reconstruction model
# fexp = 

# map1 and map2
img2text = modules.IMG2TEXTwithEXP(384*4,384*4,768)
img2text = img2text.to(device)
img2text.train()

params=list([p for p in img2text.parameters() if p.requires_grad])

optimizer_cls = torch.optim.AdamW
optimizer = optimizer_cls(
    [
        {"params": params, "lr": 1e-5},
    ],
)

lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=50,
    )


# train
for itr in range(iteration):
    train_loss = 0
    total=len(dataloader)
    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="batch train"):
        # print(batch)
        num_objects = batch["num_objects"]
        object_pixel_values = batch["object_pixel_values"]
        pos_id = batch["image_token_idx_mask"]
        pos_id_image=batch["image_token_idx"]
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        image_token_mask = batch["image_token_mask"]
        cgdr_ids=batch["cgdr_ids"]
        cgdr_parsing=batch["cgdr_parsing"]



        idvec = msid.extract_mlfeat(object_pixel_values.to(device).float(),[2,5,8,11])
        # expvec = fexp(inputs)

        # rdm = np.random.randint(10)
        # if rdm < 2:
        tokenized_identity_first, tokenized_identity_last = img2text(idvec,exp=None)
 
        hidden_states = utils.get_clip_hidden_states(input_ids.to(device),pipe.text_encoder).to(dtype=torch.float32)
        
        input_shape = input_ids.size()
        
        bsz, seq_len = input_shape

        pos_eot = input_ids.to(dtype=torch.int, device=hidden_states.device).argmax(dim=-1)

        # return_dict = pipe.text_encoder.use_return_dict

        encoder_outputs = pipe.text_encoder(
            hidden_states=hidden_states, pos_eot=pos_eot

        )[0]

        last_hidden_state = encoder_outputs[0]
        # last_hidden_state = pipe.text_encoder.final_layer_norm(last_hidden_state)

        
        encoder_hidden_states = fuse_object_embeddings(
            encoder_outputs, image_token_mask, tokenized_identity_first, num_objects
        )

        vae_input = batch['pixel_values']

        latents = pipe.vae.encode(vae_input).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents).to("cuda:0")   
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
                    0, pipe.scheduler.num_train_timesteps, (bsz,), device=latents.device
                )
        timesteps = timesteps.long().to("cuda:0")   

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps).to("cuda:0")   

        pred = unet(noisy_latents.to(dtype=torch.float32), timesteps, encoder_hidden_states.to(device="cuda:0",dtype=torch.float32)).sample



        # cgdr
        # print("cgdr_ids",cgdr_ids.shape)
       
        cgdr_hidden_states = utils.get_clip_hidden_states(cgdr_ids.to(device),pipe.text_encoder).to(dtype=torch.float32)
        
        cgdr_shape = cgdr_ids.size()
        
        bsz, seq_len = cgdr_shape
        # causal_attention_mask = pipe.text_encoder._build_causal_attention_mask(
        #     bsz, seq_len, hidden_states.dtype
        # ).to(hidden_states.device)
        cgdr_pos_eot = cgdr_ids.to(dtype=torch.int, device=cgdr_hidden_states.device).argmax(dim=-1)

        # return_dict = pipe.text_encoder.use_return_dict
        cgdr_encoder_outputs = pipe.text_encoder(
            hidden_states=cgdr_hidden_states, pos_eot=cgdr_pos_eot
        )[0]
        cgdr_last_hidden_state = cgdr_encoder_outputs[0]


        cgdr_encoder_hidden_states = fuse_object_embeddings(
            cgdr_encoder_outputs, image_token_mask, tokenized_identity_first, num_objects
        )
        cgdr_pred = unet(noisy_latents.to(dtype=torch.float32), timesteps, cgdr_encoder_hidden_states.to(device="cuda:0",dtype=torch.float32)).sample


        # mask
        # loss1=cgdr_parsing
        cgdr_parsing=cgdr_parsing.permute(0,3,1,2)
        resized_input = F.interpolate(cgdr_parsing, size=(14, 14), mode='bilinear', align_corners=False)  # 形状变为 [16, 3, 14, 14]
        resized_channel=resized_input[:,0,:,:]  # 取第一个通道
        # 将缩小后的单通道张量添加到通道维度上
        resized_channel = resized_channel.unsqueeze(1)  # 形状变为 [16, 1, 14, 14]
        cgdr_mask = torch.cat([resized_input, resized_channel], dim=1).to("cuda:0")  # 形状变为 [16, 4, 14, 14]
        cgdr_mask_minus=1-cgdr_mask

        pred_mask=pred*cgdr_mask
        cgdr_pred_mask=cgdr_pred*cgdr_mask_minus
        pred_new=pred_mask+cgdr_pred_mask

        # print(cgdr_parsing.shape)
        # print(cgdr_pred.shape)
        # print(pred.shape)


        loss = F.mse_loss(pred_new.float(), noise.float(), reduction="mean")
        # print(f"iteration:{itr}  loss:{loss}")
        train_loss += loss.item()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    PATH = f'./fmap/fmap_epoch{itr}.pth'
    torch.save(img2text.state_dict(), PATH)
    with open("./training_loss.txt", 'a') as train_los:
        train_los.write(f"{train_loss/total}\n")
    print(f"iteration:{itr}  loss:{train_loss/total}")