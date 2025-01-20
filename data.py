import os
import torch
from torchvision.io import read_image, ImageReadMode
import glob
import json
import numpy as np
from PIL import Image
import os.path as osp

import torchvision.transforms as transforms
import cv2
import random
from copy import deepcopy
from model import BiSeNet
from test import vis_parsing_maps
from transform import (
    get_train_transforms_with_segmap,
    get_object_transforms,
    get_object_processor,
)


def prepare_image_token_idx(image_token_mask, max_num_objects):
    image_token_idx = torch.nonzero(image_token_mask, as_tuple=True)[1]
    image_token_idx_mask = torch.ones_like(image_token_idx, dtype=torch.bool)
    if len(image_token_idx) < max_num_objects:
        image_token_idx = torch.cat(
            [
                image_token_idx,
                torch.zeros(max_num_objects - len(image_token_idx), dtype=torch.long),
            ]
        )
        image_token_idx_mask = torch.cat(
            [
                image_token_idx_mask,
                torch.zeros(
                    max_num_objects - len(image_token_idx_mask),
                    dtype=torch.bool,
                ),
            ]
        )

    image_token_idx = image_token_idx.unsqueeze(0)
    image_token_idx_mask = image_token_idx_mask.unsqueeze(0)
    # print(image_token_idx)
    # print(image_token_mask)
    return image_token_idx, image_token_idx_mask


class DemoDataset(object):
    def __init__(
        self,
        test_caption,
        test_reference_folder,
        tokenizer,
        object_transforms,
        image_token="<|image|>",
        max_num_objects=4,
        device=None,
    ) -> None:
        self.test_caption = test_caption
        self.test_reference_folder = test_reference_folder
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.object_transforms = object_transforms

        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.max_num_objects = max_num_objects
        self.device = device
        self.image_ids = None

    def set_caption(self, caption):
        self.test_caption = caption

    def set_reference_folder(self, reference_folder):
        self.test_reference_folder = reference_folder

    def set_image_ids(self, image_ids=None):
        self.image_ids = image_ids

    def get_data(self):
        return self.prepare_data()

    def _tokenize_and_mask_noun_phrases_ends(self, caption):
        input_ids = self.tokenizer.encode(caption)

        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    def prepare_data(self):
        object_pixel_values = []
        image_ids = []

        for image_id in self.image_ids:
            reference_image_path = sorted(
                glob.glob(os.path.join(self.test_reference_folder, image_id, "*.jpg"))
                + glob.glob(os.path.join(self.test_reference_folder, image_id, "*.png"))
                + glob.glob(
                    os.path.join(self.test_reference_folder, image_id, "*.jpeg")
                )
            )[0]

            reference_image = self.object_transforms(
                read_image(reference_image_path, mode=ImageReadMode.RGB)
            ).to(self.device)
            object_pixel_values.append(reference_image)
            image_ids.append(image_id)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            self.test_caption
        )

        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )

        num_objects = image_token_idx_mask.sum().item()

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        return {
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "num_objects": torch.tensor(num_objects),
            "filenames": image_ids,
        }


class FastComposerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        max_num_objects=4,
        num_image_tokens=1,
        image_token="<|image|>",
        object_appear_prob=1,
        uncondition_prob=0,
        text_only_prob=0,
        object_types=None,
        split="all",
        min_num_objects=None,
        balance_num_objects=False,
        cgdr=False,
    ):
        self.root = "data/ffhq_wild_files"
        self.tokenizer = tokenizer
        self.train_transforms = get_train_transforms_with_segmap()
        self.object_transforms = get_object_transforms("train")
        self.object_processor = get_object_processor()
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.num_image_tokens = num_image_tokens
        self.object_appear_prob = object_appear_prob
        self.device = "cuda"
        self.uncondition_prob = uncondition_prob
        self.text_only_prob = text_only_prob
        self.object_types = object_types

        self.cgdr=cgdr

        if split == "all":
            image_ids_path = os.path.join(root, "image_ids.txt")
        elif split == "train":
            image_ids_path = os.path.join(root, "image_ids_train.txt")
        elif split == "test":
            image_ids_path = os.path.join(root, "image_ids_test.txt")
        else:
            raise ValueError(f"Unknown split {split}")

        with open(image_ids_path, "r") as f:
            self.image_ids = f.read().splitlines()

        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)


        # bisenet
        n_classes = 19
        self.net = BiSeNet(n_classes=n_classes)
        self.net.cuda()
        self.net.load_state_dict(torch.load("res/cp/79999_iter.pth"))
        self.net.eval()


        if min_num_objects is not None:
            print(f"Filtering images with less than {min_num_objects} objects")
            filtered_image_ids = []
            for image_id in tqdm(self.image_ids):
                chunk = image_id[:5]
                info_path = os.path.join(self.root, chunk, image_id + ".json")
                with open(info_path, "r") as f:
                    info_dict = json.load(f)
                segments = info_dict["segments"]

                if self.object_types is not None:
                    segments = [
                        segment
                        for segment in segments
                        if segment["coco_label"] in self.object_types
                    ]

                if len(segments) >= min_num_objects:
                    filtered_image_ids.append(image_id)
            self.image_ids = filtered_image_ids

        if balance_num_objects:
            _balance_num_objects(self)

    def __len__(self):
        return len(self.image_ids)

    def _tokenize_and_mask_noun_phrases_ends(self, caption, segments):
        for segment in reversed(segments):
            end = segment["end"]
            caption = caption[:end] + self.image_token + caption[end:]

        input_ids = self.tokenizer.encode(caption)

        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    @torch.no_grad()
    def preprocess(self, image, info_dict, segmap, image_id, cgdr_img=None):  
        # print(image.shape)

        # img=image.cpu().numpy()
        # print(img)
        if self.cgdr:
            cgdr_img = cgdr_img.resize((512, 512), Image.BILINEAR)
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            img = to_tensor(cgdr_img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            vis_mask=vis_parsing_maps(cgdr_img, parsing, stride=1, save_im=False)
            
        
            low_res_parsing = cv2.resize(vis_mask, (112, 112), interpolation=cv2.INTER_NEAREST)
            low_res_parsing=low_res_parsing/255.0
        else:
            low_res_parsing = np.ones((112, 112), dtype=np.float32)
            # count = np.sum(low_res_parsing == 1)
        # print(count)
        # print(low_res_parsing.shape)
        # print(low_res_parsing)

        caption = info_dict["caption"]
        segments = info_dict["segments"]

        if self.object_types is not None:
            segments = [
                segment
                for segment in segments
                if segment["coco_label"] in self.object_types
            ]

        pixel_values, transformed_segmap = self.train_transforms(image, segmap)

        object_pixel_values = []
        object_segmaps = []

        prob = random.random()
        if prob < self.uncondition_prob:
            caption = ""
            segments = []
        elif prob < self.uncondition_prob + self.text_only_prob:
            segments = []
        else:
            segments = [
                segment
                for segment in segments
                if random.random() < self.object_appear_prob
            ]

        if len(segments) > self.max_num_objects:
            # random sample objects
            segments = random.sample(segments, self.max_num_objects)

        segments = sorted(segments, key=lambda x: x["end"])

        background = self.object_processor.get_background(image)

        for segment in segments:
            id = segment["id"]
            bbox = segment["bbox"]  # [h1, w1, h2, w2]
            object_image = self.object_processor(
                deepcopy(image), background, segmap, id, bbox
            )
            object_pixel_values.append(self.object_transforms(object_image))
            object_segmaps.append(transformed_segmap == id)

        # print("caption",caption)
        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            caption, segments
        )
        # cgdr_ids=self.tokenizer.encode("a photo of a person")
        if self.cgdr:
            cgdr_caption="a photo of a person"
            cgdr_ids, cgdr_image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
                cgdr_caption, segments
            )

        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )

        num_objects = image_token_idx_mask.sum().item()
        object_pixel_values = object_pixel_values[:num_objects]
        object_segmaps = object_segmaps[:num_objects]

        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values[0])
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0

        if num_objects < self.max_num_objects:
            object_pixel_values += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]
            object_segmaps += [
                torch.zeros_like(transformed_segmap)
                for _ in range(self.max_num_objects - num_objects)
            ]

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        object_segmaps = torch.stack(
            object_segmaps
        ).float()  # [max_num_objects, 256, 256]

        
        if self.cgdr:
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "image_token_mask": image_token_mask,
                "image_token_idx": image_token_idx,
                "image_token_idx_mask": image_token_idx_mask,
                "object_pixel_values": object_pixel_values,
                "object_segmaps": object_segmaps,
                "num_objects": torch.tensor(num_objects),
                "image_ids": torch.tensor(image_id),
                "cgdr_ids":torch.tensor(cgdr_ids),
                "cgdr_parsing":torch.tensor(low_res_parsing),
            }
        else:
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "image_token_mask": image_token_mask,
                "image_token_idx": image_token_idx,
                "image_token_idx_mask": image_token_idx_mask,
                "object_pixel_values": object_pixel_values,
                "object_segmaps": object_segmaps,
                "num_objects": torch.tensor(num_objects),
                "image_ids": torch.tensor(image_id),
            }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        chunk = image_id[:5]
        image_path = os.path.join(self.root, chunk, image_id + ".jpg")
        info_path = os.path.join(self.root, chunk, image_id + ".json")
        segmap_path = os.path.join(self.root, chunk, image_id + ".npy")
        if self.cgdr:
            image = read_image(image_path, mode=ImageReadMode.RGB)
            cgdr_img=Image.open(image_path)
        else:
            cgdr_img=None

        with open(info_path, "r") as f:
            info_dict = json.load(f)
        segmap = torch.from_numpy(np.load(segmap_path))

        if self.device is not None:
            image = image.to(self.device)
            segmap = segmap.to(self.device)
        
        item=self.preprocess(image, info_dict, segmap, int(image_id),cgdr_img=cgdr_img)
        # print(item)
        return item


def collate_fn(examples):
    cgdr=False
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])
    if cgdr:
        cgdr_ids=torch.cat([example["cgdr_ids"] for example in examples])
        cgdr_parsing=torch.stack([example["cgdr_parsing"] for example in examples]) 
    image_ids = torch.stack([example["image_ids"] for example in examples])


    image_token_mask = torch.cat([example["image_token_mask"] for example in examples])
    image_token_idx = torch.cat([example["image_token_idx"] for example in examples])
    image_token_idx_mask = torch.cat(
        [example["image_token_idx_mask"] for example in examples]
    )

    object_pixel_values = torch.stack(
        [example["object_pixel_values"] for example in examples]
    )
    object_segmaps = torch.stack([example["object_segmaps"] for example in examples])

    num_objects = torch.stack([example["num_objects"] for example in examples])
    if cgdr:
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "object_segmaps": object_segmaps,
            "num_objects": num_objects,
            "imag_ids": image_ids,
            "cgdr_ids":cgdr_ids,
            "cgdr_parsing":cgdr_parsing,
        }
    else:
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "object_segmaps": object_segmaps,
            "num_objects": num_objects,
            "imag_ids": image_ids,
        }

def get_data_loader(dataset, batch_size, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=0,
    )

    return dataloader
