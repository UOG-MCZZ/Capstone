import torch
from transformers import AutoProcessor
from transformers import AutoModelForTokenClassification
from transformers import LayoutLMv3ImageProcessor, LayoutLMv3TokenizerFast, LayoutLMv3Processor

label_list = ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
id2label = {0: 'O', 1: 'B-HEADER', 2: 'I-HEADER', 3: 'B-QUESTION', 4: 'I-QUESTION', 5: 'B-ANSWER', 6: 'I-ANSWER'}

model = AutoModelForTokenClassification.from_pretrained("./test/checkpoint-1000")

img_processor = LayoutLMv3ImageProcessor()
tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
my_processor = LayoutLMv3Processor(img_processor, tokenizer=tokenizer)

def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]

def runInference(image):
    encoding = my_processor(image, return_offsets_mapping=True, return_tensors="pt")

    offset_mapping = encoding.pop('offset_mapping')

    with torch.no_grad(): 
        outputs = model(**encoding)

    logits = outputs.logits
    print("pred shape\n", logits.shape)

    predictions = logits.argmax(-1).squeeze().tolist()
    # print(predictions)

    # labels = encoding.labels.squeeze().tolist()
    # print(labels)

    token_boxes = encoding.bbox.squeeze().tolist()
    width, height = image.size

    import numpy as np

    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0

    # word_list = [id2label[pred] for idx, pred in enumerate(enco) if not is_subword[idx]]
    true_predictions = [pred for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    return {"pred": true_predictions, "boxes": true_boxes}
    return true_boxes
