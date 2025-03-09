import torch
from transformers import AutoTokenizer
import pytesseract
from PIL import Image
import numpy as np
from utils import get_class_names, get_config
import itertools
from model import get_model
from omegaconf import OmegaConf

max_seq_length = 512
max_connections = 100

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

    input_ids = np.ones(max_seq_length, dtype=np.long) * pad_token_id
    bbox = np.zeros((max_seq_length, 8), dtype=np.float32)
    attention_mask = np.zeros(max_seq_length, dtype=np.long)

    are_box_first_tokens = np.zeros(max_seq_length, dtype=np.bool_)
    el_labels = np.ones(max_seq_length, dtype=np.long) * max_seq_length

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
            # print(cum_token_idx, "link to", word_idx, "is", processed_data[word_idx]["word"], "shld be", word["word"])

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

    input_ids = torch.from_numpy(input_ids).unsqueeze(0).type(torch.LongTensor).to(torch.device("cuda:0"))
    bbox = torch.from_numpy(bbox).unsqueeze(0).to(torch.device("cuda:0"))
    attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).type(torch.LongTensor).to(torch.device("cuda:0"))

    are_box_first_tokens = torch.from_numpy(are_box_first_tokens).unsqueeze(0).type(torch.LongTensor).to(torch.device("cuda:0"))
    el_labels = torch.from_numpy(el_labels).unsqueeze(0).type(torch.LongTensor).to(torch.device("cuda:0"))

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

def get_eval_kwargs_spade(dataset_root_path, max_seq_length):
    class_names = get_class_names(dataset_root_path)
    dummy_idx = max_seq_length

    eval_kwargs = {"class_names": class_names, "dummy_idx": dummy_idx}

    return eval_kwargs

def prepare_spade(processed_data, img: Image, tokenizer: AutoTokenizer):

    width, height = img.size

    pad_token_id = tokenizer.vocab["[PAD]"]
    cls_token_id = tokenizer.vocab["[CLS]"]
    sep_token_id = tokenizer.vocab["[SEP]"]
    unk_token_id = tokenizer.vocab["[UNK]"]

    input_ids = np.ones(max_seq_length, dtype=int) * pad_token_id
    bbox = np.zeros((max_seq_length, 8), dtype=np.float32)
    attention_mask = np.zeros(max_seq_length, dtype=int)

    itc_labels = np.zeros(max_seq_length, dtype=int)
    are_box_first_tokens = np.zeros(max_seq_length, dtype=np.bool_)

    # stc_labels stores the index of the previous token.
    # A stored index of max_seq_length (512) indicates that
    # this token is the initial token of a word box.
    stc_labels = np.ones(max_seq_length, dtype=np.int64) * max_seq_length

    list_tokens = []
    list_bbs = []

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

        list_tokens += tokens

        # min, max clipping
        for coord_idx in range(4):
            bb[coord_idx][0] = max(0.0, min(bb[coord_idx][0], width))
            bb[coord_idx][1] = max(0.0, min(bb[coord_idx][1], height))

        bb = list(itertools.chain(*bb))
        bbs = [bb for _ in range(len(tokens))]

        for _ in tokens:
            cum_token_idx += 1
            token_indices_to_wordidx[cum_token_idx] = word_idx
            this_box_token_indices.append(cum_token_idx)

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

    # Normalize bbox -> 0 ~ 1
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

    # Label
    # classes_dic = json_obj["parse"]["class"]
    # for class_name in class_names:
    #     if class_name == "others":
    #         continue
    #     if class_name not in classes_dic:
    #         continue

    #     for word_list in classes_dic[class_name]:
    #         is_first, last_word_idx = True, -1
    #         for word_idx in word_list:
    #             if word_idx >= len(box_to_token_indices):
    #                 break
    #             box2token_list = box_to_token_indices[word_idx]
    #             for converted_word_idx in box2token_list:
    #                 if converted_word_idx >= max_seq_length:
    #                     break  # out of idx

    #                 if is_first:
    #                     itc_labels[converted_word_idx] = class_idx_dic[
    #                         class_name
    #                     ]
    #                     is_first, last_word_idx = False, converted_word_idx
    #                 else:
    #                     stc_labels[converted_word_idx] = last_word_idx
    #                     last_word_idx = converted_word_idx

    input_ids = torch.from_numpy(input_ids).unsqueeze(0).to(torch.device("cuda:0"))
    bbox = torch.from_numpy(bbox).unsqueeze(0).to(torch.device("cuda:0"))
    attention_mask = torch.from_numpy(attention_mask).unsqueeze(0).to(torch.device("cuda:0"))

    itc_labels = torch.from_numpy(itc_labels).unsqueeze(0).to(torch.device("cuda:0"))
    are_box_first_tokens = torch.from_numpy(are_box_first_tokens).unsqueeze(0).to(torch.device("cuda:0"))
    stc_labels = torch.from_numpy(stc_labels).unsqueeze(0).to(torch.device("cuda:0"))

    return_dict = {
        "input_ids": input_ids,
        "bbox": bbox,
        "attention_mask": attention_mask,
        "itc_labels": itc_labels,
        "are_box_first_tokens": are_box_first_tokens,
        "stc_labels": stc_labels,
        "t2w": token_indices_to_wordidx,
        "data": processed_data,
    }

    return return_dict

def parse_initial_words(itc_label, box_first_token_mask, class_names):
    itc_label_np = itc_label.cpu().numpy()
    box_first_token_mask_np = box_first_token_mask.cpu().numpy()

    outputs = [[] for _ in range(len(class_names))]
    for token_idx, label in enumerate(itc_label_np):
        if box_first_token_mask_np[token_idx] and label != 0:
            outputs[label].append(token_idx)

    return outputs

def parse_subsequent_words(stc_label, attention_mask, init_words, dummy_idx):
    valid_stc_label = stc_label * attention_mask.bool()
    valid_stc_label = valid_stc_label.cpu().numpy()
    stc_label_np = stc_label.cpu().numpy()

    valid_token_indices = np.where(
        (valid_stc_label != dummy_idx) * (valid_stc_label != 0)
    )

    next_token_idx_dict = {}
    for token_idx in valid_token_indices[0]:
        next_token_idx_dict[stc_label_np[token_idx]] = token_idx

    outputs = []
    for init_token_indices in init_words:
        sub_outputs = []
        for init_token_idx in init_token_indices:
            cur_token_indices = [init_token_idx]
            for _ in range(max_connections):
                if cur_token_indices[-1] in next_token_idx_dict:
                    if (
                        next_token_idx_dict[cur_token_indices[-1]]
                        not in init_token_indices
                    ):
                        cur_token_indices.append(
                            next_token_idx_dict[cur_token_indices[-1]]
                        )
                    else:
                        break
                else:
                    break
            sub_outputs.append(tuple(cur_token_indices))

        outputs.append(sub_outputs)

    return outputs

def get_rel(image: Image):
    image_data = prepare_image(image, net.tokenizer)
    input_dict = prepare_spade_rel(image_data, image, net.tokenizer)
    print(input_dict['data'])

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
            # link.append((t2wi[i], t2wi[idx]))
            link.append((i, idx))

    # for l in link:
    #     print(data[l[0]]["word"], ":", data[l[1]]["word"])
    #     boxes.append((data[l[0]]["bbox"], data[l[1]]["bbox"]))
    #     key_val.append((data[l[0]]["word"], data[l[1]]["word"]))
    
    input_dict = prepare_spade(image_data, image, net_ee.tokenizer)

    # Run inference
    with torch.no_grad():
        outputs = net_ee(input_dict)

    data = input_dict["data"]
    t2wi = input_dict["t2w"]
    box_first_token_mask = input_dict["are_box_first_tokens"][0]
    attention_mask = input_dict["attention_mask"][0]
    
    itc_outputs = outputs[0]["itc_outputs"]
    stc_outputs = outputs[0]["stc_outputs"]

    pr_itc_labels = torch.argmax(itc_outputs, -1)
    pr_stc_labels = torch.argmax(stc_outputs, -1)

    eval_kwargs = get_eval_kwargs_spade(cfg_ee.dataset_root_path, cfg_ee.train.max_seq_length)
    class_names = eval_kwargs["class_names"]
    dummy_idx = eval_kwargs["dummy_idx"]

    pr_init_words = parse_initial_words(pr_itc_labels[0], box_first_token_mask, class_names)
    pr_class_words = parse_subsequent_words(pr_stc_labels[0], attention_mask, pr_init_words, dummy_idx)
    
    texts = []
    block_bboxes = []
    class_info = []
    i = 0
    sex_list = [-1] * max_seq_length
    for c_idx, c in enumerate(pr_class_words):
        print("class: ", class_names[c_idx])
        for word_list in c:
            text = ""
            block_bbox = None
            for w_indice in word_list:
                if box_first_token_mask[w_indice]:
                    text += " " + data[t2wi[w_indice]]["word"]
                    if block_bbox is None: block_bbox = data[t2wi[w_indice]]["bbox"]
                    block_bbox = max_bbox(block_bbox, data[t2wi[w_indice]]["bbox"])
                sex_list[w_indice] = i
            # text = net_ee.tokenizer.decode(input_dict["input_ids"][0][list(word_list)])
            texts.append(text)
            block_bboxes.append(block_bbox)
            class_info.append(c_idx)
            i += 1
            print(text)
    print("\n\nLinking Results")
    print(pr_el_labels)

    for l in link:
        # print(data[l[0]]["word"], ":", data[l[1]]["word"])
        kv = (texts[sex_list[l[0]]], texts[sex_list[l[1]]])
        if kv not in key_val:
            boxes.append((block_bboxes[sex_list[l[0]]], block_bboxes[sex_list[l[1]]]))
            key_val.append(kv)

    ocr_boxes = [d["bbox"] for d in image_data]
    return {"link_boxes": boxes, "key_val": key_val, "block_bboxes": block_bboxes, "text": texts, "classes": class_info, "ocr_boxes": ocr_boxes}

def max_bbox(bbox1, bbox2):
    r_bbox= bbox1
    r_bbox[0] = [min(bbox1[0][0], bbox2[0][0]), min(bbox1[0][1], bbox2[0][1])]
    r_bbox[1] = [max(bbox1[1][0], bbox2[1][0]), min(bbox1[1][1], bbox2[1][1])]
    r_bbox[2] = [max(bbox1[2][0], bbox2[2][0]), max(bbox1[2][1], bbox2[2][1])]
    r_bbox[3] = [min(bbox1[3][0], bbox2[3][0]), max(bbox1[3][1], bbox2[3][1])]
    return r_bbox

def main():
    # Load the image
    image_path = r".\datasets\funsd_spade\training_data\images\93455715.png"
    image = Image.open(image_path)
    print(get_rel(image))

cfg = {'workspace': 'S:/bros/finetune_funsd_el_spade__bros-large-uncased', 'dataset': 'funsd', 'task': 'el', 'dataset_root_path': 'S:/bros/datasets/funsd_spade', 'pretrained_model_path': 'S:/bros/pretrained_models', 'seed': 1, 'cudnn_deterministic': False, 'cudnn_benchmark': True, 'model': {'backbone': 'naver-clova-ocr/bros-large-uncased', 'head': 'spade_rel', 'head_hidden_size': 128, 'n_classes': 3, 'head_p_dropout': 0.1}, 'train': {'batch_size': 4, 'num_samples_per_epoch': 149, 'max_epochs': 100, 'use_fp16': True, 'accelerator': 'gpu', 'strategy': {'type': 'ddp'}, 'clip_gradient_algorithm': 'norm', 'clip_gradient_value': 1.0, 'num_workers': 11, 'optimizer': {'method': 'adamw', 'params': {'lr': 5e-05}, 'lr_schedule': {'method': 'linear', 'params': {'warmup_steps': 0}}}, 'val_interval': 1, 'max_seq_length': 512}, 'val': {'batch_size': 1, 'num_workers': 4, 'limit_val_batches': 1.0}, 'pretrained_model_file': 'S:/bros/checkpoints/large.ckpt', 'save_weight_dir': 'S:/bros/finetune_funsd_el_spade__bros-large-uncased\\checkpoints', 'tensorboard_dir': 'S:/bros/finetune_funsd_el_spade__bros-large-uncased\\tensorboard_logs'}
cfg = OmegaConf.create(cfg)
print(cfg)

net = get_model(cfg)

load_model_weight(net, cfg.pretrained_model_file)

net.to("cuda")
net.eval()

cfg_ee = {'workspace': 'S:/bros/finetune_funsd_ee_spade__bros-base-uncased', 'dataset': 'funsd', 'task': 'ee', 'dataset_root_path': 'S:/bros/datasets/funsd_spade', 'pretrained_model_path': 'S:/bros/pretrained_models', 'seed': 1, 'cudnn_deterministic': False, 'cudnn_benchmark': True, 'model': {'backbone': 'naver-clova-ocr/bros-base-uncased', 'head': 'spade', 'head_hidden_size': 128, 'n_classes': 3, 'head_p_dropout': 0.1}, 'train': {'batch_size': 4, 'num_samples_per_epoch': 149, 'max_epochs': 100, 'use_fp16': True, 'accelerator': 'gpu', 'strategy': {'type': 'ddp'}, 'clip_gradient_algorithm': 'norm', 'clip_gradient_value': 1.0, 'num_workers': 11, 'optimizer': 
{'method': 'adamw', 'params': {'lr': 5e-05}, 'lr_schedule': {'method': 'linear', 'params': {'warmup_steps': 0}}}, 'val_interval': 1, 'max_seq_length': 512}, 'val': {'batch_size': 4, 'num_workers': 4, 'limit_val_batches': 1.0}, 'pretrained_model_file': 'S:/bros/checkpoints/ee.ckpt', 'save_weight_dir': 'S:/bros/finetune_funsd_ee_spade__bros-base-uncased\\checkpoints', 'tensorboard_dir': 'S:/bros/finetune_funsd_ee_spade__bros-base-uncased\\tensorboard_logs'}
cfg_ee = OmegaConf.create(cfg_ee)

net_ee = get_model(cfg_ee)

load_model_weight(net_ee, cfg_ee.pretrained_model_file)

net_ee.to("cuda")
net_ee.eval()

if __name__ == "__main__":
    main()