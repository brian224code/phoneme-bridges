import numpy as np
import pandas as pd
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, AutoTokenizer, AutoModel, utils
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch


def convert(sentences, tags):
  for i in range(len(tags)):
    text = sentences[i]
    entities = tags[i].split()
    new_entities = []
    start = 0
    for j, word in enumerate(text.split()):
      new_entities.append((start, start + len(word), entities[j]))
      start += len(word) + 1
    tags[i] = new_entities

entity_types = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

num_labels = len(entity_types)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', device_map=device)
# Move model to the appropriate device
# model_name = "cge7/udpos-hindi-urdu-romanization"
# model_name = "cge7/udpos-urdu-romanization"
model_names = ["cge7/udpos-urdu-romanization", "cge7/udpos-hindi-urdu-romanization"]
models = []
hf_token = PASTE_YOUR_TOKEN_HERE

for model_name in model_names:
  models.append(BertForTokenClassification.from_pretrained(model_name, token=hf_token, output_attentions=True).to(device))


def convert_with_romanization(sentences, romanizations, pos_tags):
    new_data = []
    new_tags = []

    for sentence, romanization, tags in zip(
        sentences, romanizations, pos_tags
    ):
        new_text = []
        new_entities = []
        start = 0

        sentence = sentence.split()
        romanization = romanization.split()
        tags = tags.split()

        for word, tag in zip(sentence, tags):
            new_text.append(word)
            end = start + len(word)
            new_entities.append((start, end, tag, 1))
            start = end + 1

        for word, tag in zip(romanization, tags):
            new_text.append(word)
            end = start + len(word)
            new_entities.append((start, end, tag, 0))
            start = end + 1

        new_tags.append(new_entities)
        new_text_str = " ".join(new_text)
        new_data.append(new_text_str)

    return new_data, new_tags

def format_with_romanization(sentences, tags):
    tokenized_data = []
    for i in range(len(sentences)):
        text = sentences[i]
        entities = tags[i]  # List of (start, end, entity_type, not_roman)

        # Tokenize with offsets
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding='max_length'
        )
        input_ids = encoding['input_ids']
        offsets = encoding['offset_mapping']

        labels = ['ADJ'] * len(input_ids)
        importance = [0] * len(input_ids)
        is_start = [False] * len(input_ids)

        for start_char, end_char, entity_type, not_roman in entities:
            for idx, (offset_start, offset_end) in enumerate(offsets):
                if offset_start is None or offset_end is None:
                    continue  # Skip special tokens
                if offset_end <= start_char:
                    continue
                if offset_start >= end_char:
                    break
                if offset_start >= start_char and offset_end <= end_char:
                    labels[idx] = entity_type
                    importance[idx] = not_roman
                    if offset_start == start_char:
                        is_start[idx] = True

        label_ids = [entity_types.index(label) for label in labels]

        tokenized_data.append({
            'input_ids': input_ids,
            'labels': label_ids,
            'importance': importance,
            'is_start': is_start
        })
    return tokenized_data



original_evalset = pd.read_csv("dev-ur-romanized.csv")

eval_sentences = original_evalset['Sentence'].tolist()
eval_romanizations = original_evalset['Romanization'].tolist()
eval_tags = original_evalset['Tags'].tolist()
eval_sentences, eval_tags = convert_with_romanization(eval_sentences, eval_romanizations, eval_tags)
tokenized_data = format_with_romanization(eval_sentences, eval_tags)

from bertviz import head_view
from sklearn.metrics import f1_score

def evaluate_f1(tokenized_data, eval_sentences):
    y_pred, y_true = [], []

    for i in range(2, 20):
        text = eval_sentences[i]
        inputs = tokenizer(text, return_tensors='pt')
        inputs = inputs.to(device)
        print(f"{i=}")
        with torch.no_grad():
            for i, model in enumerate(models):
              outputs = model(**inputs)
              attention = outputs[-1]
              token_ids = inputs["input_ids"][0]  # Extract input_ids from the batch
              tokens = tokenizer.convert_ids_to_tokens(token_ids)  # Convert to tokens
              print(f"{len(tokens)=}")
              if len(tokens) > 80:
                break
              print(f"{model_names[i]}")
              head_view(attention, tokens)  # Visualize attention with BertViz

            # outputs = model_hindi_urdu(**inputs)
            # attention = outputs[-1]
            # token_ids = inputs["input_ids"][0]  # Extract input_ids from the batch
            # tokens = tokenizer.convert_ids_to_tokens(token_ids)  # Convert to tokens
            # print(f"{tokens=}")
            # print(f"{len(tokens)=}")
            # head_view(attention, tokens)  # Visualize attention with BertViz


        # predicted_labels = outputs.logits.argmax(dim=-1).tolist()[0]
        # token_labels = tokenized_data[i]['labels']
        # token_importance = tokenized_data[i]['importance']
        # is_start = tokenized_data[i]['is_start']

        # # Extract the first token of each word using importance and its preceding token
        # for j, importance_flag in enumerate(token_importance):
        #     if importance_flag == 1 and is_start[j]:  # First token logic
        #         y_true.append(token_labels[j])
        #         y_pred.append(predicted_labels[j])

    return y_true, y_pred




# Evaluate model
y_true, y_pred = evaluate_f1(tokenized_data, eval_sentences)
