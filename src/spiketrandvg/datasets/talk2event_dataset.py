import os
from pathlib import Path
from torch.utils.data import Dataset
import json
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pickle
import json
from PIL import Image
# from utils import strefer_utils, pc_utils
# from tqdm import tqdm
from transformers import RobertaTokenizerFast
import torchvision.transforms.functional as tv_F

import random

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

CLASSES = ('pedestrian', 'rider', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'train')

# Upstream hardcoded SRC_PATH = "/dataset/shared/magiclidar/" (the authors' machine).
# Resolved here from T2E_DATA_ROOT, else the dataset/ dir of this repo's workspace.
# Must end in a separator: it is used with os.path.join on already-relative paths.
SRC_PATH = os.environ.get(
    "T2E_DATA_ROOT",
    str(Path(__file__).resolve().parents[4] / "dataset" / "talk2event"),
).rstrip("/") + "/"
PIXEL_MEAN = [123.675, 116.280, 103.530]
PIXEL_STD = [58.395, 57.120, 57.375]

# what upstream's Normalize actually used (torchvision convention, on [0,1] tensors)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Talk2EventDataset(Dataset):
    def __init__(self, args, image_set="train") -> None:
        super().__init__()

        print("Initializing Talk2EventDataset")
        self.attribute = args.attribute

        self.datasize = [480,640]
        #path for meta data
        meta_data_path = os.path.join(SRC_PATH, 'meta_data_v10', image_set)

        #sequence list
        self.dataset = []

        sequence_list = os.listdir(meta_data_path)
        miss_attributes = 0
        false_match_count = 0
        for sequence in sorted(sequence_list):

            #load meta data
            meta_data = json.load(open(os.path.join(meta_data_path, sequence)))

            for data_item in meta_data:

                for idx in range(len(data_item['captions'])):
                    item = {}

                    item['id'] = data_item['id']
                    item['image_path'] = os.path.join(SRC_PATH, data_item['image_path'].replace('.jpg', '.png'))
                    item['event_path'] = os.path.join(SRC_PATH, data_item['event_path'])
                    item['bbox'] = data_item['bbox']
                    item['class'] = data_item['class']
                    item['other_num_objects'] = data_item['number of other objects']

                    if self.attribute == 'all' or self.attribute == 'fusion':
                        #use all attributes
                        item['caption'] = data_item['captions'][idx]
                        item['attributes'] = data_item['attributes'][idx]              
                        self.dataset.append(item)  

                    else:         
                        #use specific attribute, need to check if attribute can be matched, and skip empty attributes

                        caption = data_item['captions'][idx].lower()

                        if len(data_item['attributes'][idx][self.attribute])==0:
                            miss_attributes += 1

                        else:
                            caption = " ".join(caption.replace(",", " ,").replace(".", " .").split())  + ". not mentioned"

                            for attribute in data_item['attributes'][idx][self.attribute]:

                                attribute = attribute.lower()
                                attribute = " ".join(attribute.replace(",", " ,").replace(".", " .").split())

                                _, _, matched_phrase = find_fuzzy_span(caption, attribute)

                                if matched_phrase is not None:
                                    item['caption'] = data_item['captions'][idx]
                                    item['attributes'] = data_item['attributes'][idx]  
                                    self.dataset.append(item)
                                    break

                                else:
                                    false_match_count += 1

        print(f"missing {miss_attributes} attributes")
        print(f"load {len(self.dataset)} data from {image_set} split")
        print(f"false match {false_match_count} attributes")

        #读取RoBERTa的tokenizer, 用于处理文本数据
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

        #image normalizer
        # pixel_mean = np.array(PIXEL_MEAN).reshape(3, 1, 1).astype(np.float32)
        # pixel_std = np.array(PIXEL_STD).reshape(3, 1, 1).astype(np.float32)
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std

    def __getitem__(self, index, custom_aug=False):
        data = self.dataset[index]

        #load image
        image_path = data["image_path"]
        image = Image.open(image_path).convert("RGB")
        H,W = self.datasize

        # #load event
        event_path = data["event_path"]
        event_data = np.load(event_path)
        event = torch.from_numpy(event_data['events'].astype(np.float32))
        # event = event.unsqueeze(1)
        # x_down = F.interpolate(event, size=(256, 320), mode='bilinear', align_corners=False)
        # event = x_down.squeeze(1)

        #load bbox
        bbox = data["bbox"]

        #change bbox to x, y, x_end, y_end
        x_end = (bbox['x'] + bbox['w'])
        y_end = (bbox['y'] + bbox['h'])
        gt_box = torch.as_tensor([
            bbox['x'], bbox['y'], x_end, y_end], dtype=torch.float32)
        gt_box[0::2].clamp_(0, W)
        gt_box[1::2].clamp_(0, H)
        gt_box = gt_box.unsqueeze(0)

        #load bbox class
        bbox_class = data["class"]
        bbox_class = CLASSES.index(bbox_class)
        bbox_class = torch.as_tensor(bbox_class, dtype=torch.int64)
        bbox_class = bbox_class.unsqueeze(0)

        #load caption
        caption = data["caption"].lower()
        caption = " ".join(caption.replace(",", " ,").replace(".", " .").split())  + ". not mentioned"

        #load attributes
        tokens_positive_list, tokens_positive = [], []

        if self.attribute == 'all':

            for attribute, anno in data["attributes"].items():

                if len(anno) == 0:
                    continue
                for i in range(len(anno)):
                    attribute = anno[i].lower()
                    attribute = " ".join(attribute.replace(",", " ,").replace(".", " .").split())
                    start, end, _ = find_fuzzy_span(caption, attribute)
                    if start is None:
                        continue
                    elif start<0:
                        start = 0
                    tokens_positive.append((start,end)) 

            tokens_positive_list.append(tokens_positive)

        elif self.attribute == 'fusion':

            for attribute, anno in data["attributes"].items():
                tokens_positive = []

                if len(anno) == 0:
                    tokens_positive_list.append(tokens_positive)
                    continue
                for i in range(len(anno)):
                    attribute = anno[i].lower()
                    attribute = " ".join(attribute.replace(",", " ,").replace(".", " .").split())
                    start, end, _ = find_fuzzy_span(caption, attribute)
                    if start is None:
                        # tokens_positive.append([]) 
                        continue
                    if start<0:
                        start = 0
                    tokens_positive.append((start,end)) 

                tokens_positive_list.append(tokens_positive)

        else:

            for attribute in data["attributes"][self.attribute]:
                    
                attribute = attribute.lower()
                attribute = " ".join(attribute.replace(",", " ,").replace(".", " .").split())
                start, end, _ = find_fuzzy_span(caption, attribute)
                if start is None:
                    continue
                if start<0:
                    start = 0
                tokens_positive.append((start,end))

            tokens_positive_list.append(tokens_positive)

        tokenized = self.tokenizer(caption, return_tensors="pt")
        positive_map = create_positive_map(tokenized, tokens_positive_list)
            
        target = {}
        target["boxes"] = gt_box
        target["labels"] = bbox_class
        target["caption"] = caption
        target["tokens_positive"] = tokens_positive_list
        target["positive_map"] = positive_map
        target["orig_size"] = torch.as_tensor([int(H), int(W)])
        target["size"] = torch.as_tensor([int(H), int(W)])
        target["category"] = data['class']
        target["other_num_objects"] = data["other_num_objects"]
        target['image_path'] = data['image_path']
        target['event_path'] = data['event_path']
        target['attributes'] = data['attributes']


        # NO AUGMENTATION. Upstream ran Compose([ToTensor, Normalize]) here, with every
        # geometric aug either commented out or disabled by cautious=True, so train and
        # test were already identical. The two operations that actually did anything are
        # inlined below; behaviour is unchanged, the intent is now explicit.
        image = tv_F.to_tensor(image)                                   # PIL -> (3,H,W) in [0,1]
        image = tv_F.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # boxes: absolute xyxy -> normalised cxcywh. Upstream divided by the IMAGE
        # tensor's h/w; we divide by this sample's own H/W so the event branch can be
        # resized independently without silently desynchronising the boxes.
        x0, y0, x1, y1 = target["boxes"].unbind(-1)
        target["boxes"] = torch.stack(
            [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H], dim=-1
        )

        input = {}
        input["image"] = image
        input["event"] = event

        return input, target

    def __len__(self):
        return len(self.dataset)


def save_pkl(file, output_path):
    output = open(output_path, "wb")
    pickle.dump(file, output)
    output.close()

def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            beg_pos = tokenized.char_to_token(beg)
            end_pos = tokenized.char_to_token(end - 1)
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos : end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


from rapidfuzz import fuzz, process

import re

def find_fuzzy_span(caption, label):
    caption_words = re.findall(r'\w+(?:-\w+)?', caption)  # include hyphenated words
    best_score = 0
    best_phrase = ""
    window_size = len(label.split())  # window size is the length of the label
    # Slide a window over the caption to compare n-grams
    for i in range(len(caption_words) - window_size + 1):
        for j in range(window_size-2, window_size + 2):
            span_words = caption_words[i:i+j]
            phrase = " ".join(span_words)
            score = fuzz.ratio(label, phrase)
            if score > best_score:
                best_score = score
                best_phrase = phrase
    
    if best_score > 70:  # threshold can be tuned
        start = caption.find(best_phrase)
        end = start + len(best_phrase)
        return start, end, best_phrase
    else:
        return None, None, None

