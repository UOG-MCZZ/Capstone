import json
import torch
from transformers import AutoTokenizer
from model import BROSSPADERELModel
import pytesseract
from PIL import Image
import numpy as np
from utils import get_class_names, get_config
import itertools
from model import get_model

max_seq_length = 512

def load_model_weight(net, pretrained_model_file):
    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")[
        "state_dict"
    ]
    new_state_dict = {}
    for k, v in pretrained_model_state_dict.items():
        new_k = k
        if new_k.startswith("net."):
            new_k = new_k[len("net.") :]
        new_state_dict[new_k] = v
    net.load_state_dict(new_state_dict)

def prepare_image(img: Image, tokenizer: AutoTokenizer):
    # Perform OCR with detailed output
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    # Extract words, confidence, and bounding boxes
    words = ocr_data["text"]
    confidences = ocr_data["conf"]
    # bounding_boxes = list(
    #     zip(ocr_data["left"], ocr_data["top"], ocr_data["width"], ocr_data["height"])
    # )
    bounding_boxes = [[[left, top],
           [left + width, top],
           [left + width, top + height],
           [left, top + height]
        ] for left, top, width, height in zip(ocr_data["left"], ocr_data["top"], ocr_data["width"], ocr_data["height"])]
    # Filter out empty words
    processed_data = [
        {"word": word, "confidence": conf, "bbox": bbox, "tokens": tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))}
        for word, conf, bbox in zip(words, confidences, bounding_boxes)
        if word.strip()
    ]
    return processed_data

def prepare_spade_rel(processed_data, img: Image, tokenizer: AutoTokenizer):

    width, height = img.size


    pad_token_id = tokenizer.vocab["[PAD]"]
    cls_token_id = tokenizer.vocab["[CLS]"]
    sep_token_id = tokenizer.vocab["[SEP]"]
    unk_token_id = tokenizer.vocab["[UNK]"]

    input_ids = np.ones(max_seq_length, dtype=int) * pad_token_id
    bbox = np.zeros((max_seq_length, 8), dtype=np.float32)
    attention_mask = np.zeros(max_seq_length, dtype=int)

    are_box_first_tokens = np.zeros(max_seq_length, dtype=np.bool_)
    el_labels = np.ones(max_seq_length, dtype=int) * max_seq_length

    list_tokens = []
    list_bbs = []
    box2token_span_map = []

    box_to_token_indices = []
    cum_token_idx = 0

    cls_bbs = [0.0] * 8

    token_indices_to_wordidx = [-1] * max_seq_length

    for word_idx, word in enumerate(processed_data):
        this_box_token_indices = []

        tokens = word["tokens"]
        bb = word["bbox"]
        if len(tokens) == 0:
            tokens.append(unk_token_id)

        if len(list_tokens) + len(tokens) > max_seq_length - 2:
            break

        box2token_span_map.append(
            [len(list_tokens) + 1, len(list_tokens) + len(tokens) + 1]
        )  # including st_idx
        list_tokens += tokens

        # min, max clipping
        for coord_idx in range(4):
            bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
            bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

        bb = list(itertools.chain(*bb))
        bbs = [bb for _ in range(len(tokens))]

        for _ in tokens:
            cum_token_idx += 1
            this_box_token_indices.append(cum_token_idx)
            token_indices_to_wordidx[cum_token_idx] = word_idx
            print(cum_token_idx, "link to", word_idx, "is", processed_data[word_idx]["word"], "shld be", word["word"])

        list_bbs.extend(bbs)
        box_to_token_indices.append(this_box_token_indices)

    sep_bbs = [width, height] * 4

    # For [CLS] and [SEP]
    list_tokens = (
        [cls_token_id]
        + list_tokens[: max_seq_length - 2]
        + [sep_token_id]
    )
    if len(list_bbs) == 0:
        # When len(json_obj["words"]) == 0 (no OCR result)
        list_bbs = [cls_bbs] + [sep_bbs]
    else:  # len(list_bbs) > 0
        list_bbs = [cls_bbs] + list_bbs[: max_seq_length - 2] + [sep_bbs]

    len_list_tokens = len(list_tokens)
    input_ids[:len_list_tokens] = list_tokens
    attention_mask[:len_list_tokens] = 1

    bbox[:len_list_tokens, :] = list_bbs

    # bounding box normalization -> [0, 1]
    bbox[:, [0, 2, 4, 6]] = bbox[:, [0, 2, 4, 6]] / width
    bbox[:, [1, 3, 5, 7]] = bbox[:, [1, 3, 5, 7]] / height

    # if backbone_type == "layoutlm":
    #     bbox = bbox[:, [0, 1, 4, 5]]
    #     bbox = bbox * 1000
    #     bbox = bbox.astype(int)

    st_indices = [
        indices[0]
        for indices in box_to_token_indices
        if indices[0] < max_seq_length
    ]
    are_box_first_tokens[st_indices] = True

    # Label None to check
    # relations = json_obj["parse"]["relations"]
    # for relation in relations:
    #     if relation[0] >= len(box2token_span_map) or relation[1] >= len(
    #         box2token_span_map
    #     ):
    #         continue
    #     if (
    #         box2token_span_map[relation[0]][0] >= max_seq_length
    #         or box2token_span_map[relation[1]][0] >= max_seq_length
    #     ):
    #         continue

    #     word_from = box2token_span_map[relation[0]][0]
    #     word_to = box2token_span_map[relation[1]][0]
    #     el_labels[word_to] = word_from

    input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(torch.device("cuda:0"))
    bbox = torch.from_numpy(bbox).unsqueeze(0).to(torch.device("cuda:0"))
    attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).to(torch.device("cuda:0"))

    are_box_first_tokens = torch.from_numpy(are_box_first_tokens).unsqueeze(0).to(torch.device("cuda:0"))
    el_labels = torch.from_numpy(el_labels).unsqueeze(0).to(torch.device("cuda:0"))

    return_dict = {
        "input_ids": input_ids,
        "bbox": bbox,
        "attention_mask": attention_mask,
        "are_box_first_tokens": are_box_first_tokens,
        "el_labels": el_labels,
        "t2w": token_indices_to_wordidx,
        "data": processed_data,
    }

    return return_dict

def get_rel(image: Image):
    image_data = prepare_image(image, net.tokenizer)
    input_dict = prepare_spade_rel(image_data, image, net.tokenizer)

    # Run inference
    with torch.no_grad():
        outputs = net(input_dict)


    data = input_dict["data"]
    t2wi = input_dict["t2w"]
    pr_el_labels = torch.argmax(outputs[0]["el_outputs"], -1)
    # print(pr_el_labels.shape)
    # print(pr_el_labels)
    link = []
    boxes = []
    key_val = []
    for idx, i in enumerate(pr_el_labels[0]):
        if i != max_seq_length:
            link.append((t2wi[i], t2wi[idx]))

    for l in link:
        print(data[l[0]]["word"], ":", data[l[1]]["word"])
        boxes.append((data[l[0]]["bbox"], data[l[1]]["bbox"]))
        key_val.append((data[l[0]]["word"], data[l[1]]["word"]))
    return {"link_boxes": boxes, "key_val": key_val}

def main():

    # Load the image
    image_path = r".\datasets\funsd_spade\training_data\images\93455715.png"
    image = Image.open(image_path)
    get_rel(image)

cfg = get_config()
print(cfg)

net = get_model(cfg)

load_model_weight(net, cfg.pretrained_model_file)

net.to("cuda")
net.eval()