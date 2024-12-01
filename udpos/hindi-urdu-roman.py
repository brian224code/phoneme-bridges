import pandas as pd
original_dataset = pd.read_csv('train-hi-romanized.csv')

train_sentences = original_dataset['Mixed'].tolist()
train_tags = original_dataset['Tags'].tolist()

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

convert(train_sentences, train_tags)

from transformers import BertTokenizerFast, BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch

entity_types = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
num_labels = len(entity_types)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', device_map=device)
model = BertForTokenClassification.from_pretrained('bert-base-multilingual-cased', num_labels=num_labels, device_map=device)

batch_size = 32
learning_rate = 5e-5
num_epochs = 40

def format(sentences, tags):
  tokenized_data = []
  for i in range(len(sentences)):
    text = sentences[i]
    entities = tags[i]
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    labels = ['ADJ'] * len(tokens)
    for start, end, entity_type in entities:
      prefix_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text[:start])))[:-1]
      start_token = len(prefix_tokens)
      entity_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text[start:end])))[1:-1]
      end_token = start_token + len(entity_tokens)
      for j in range(start_token, end_token):
        labels[j] = entity_type
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    label_ids = [entity_types.index(label) for label in labels]
    padding_length = tokenizer.model_max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    label_ids += [entity_types.index('ADJ')] * padding_length
    tokenized_data.append({'input_ids': [input_ids[i] for i in range(len(input_ids))], 'labels': [label_ids[i] for i in range(len(label_ids))]})
  return tokenized_data

tokenized_data = format(train_sentences, train_tags)

train_data = TensorDataset(
  torch.tensor([item['input_ids'] for item in tokenized_data]),
  torch.tensor([item['labels'] for item in tokenized_data])
)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
  model.train()
  for batch in tqdm(train_loader, desc='Training'):
    inputs, labels = batch
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

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

original_dataset = pd.read_csv('train-ur-romanized.csv')

train_sentences = original_dataset['Sentence'].tolist()
train_romanizations = original_dataset['Romanization'].tolist()
train_tags = original_dataset['Tags'].tolist()

train_sentences, train_tags = convert_with_romanization(train_sentences, train_romanizations, train_tags)
tokenized_data = format_with_romanization(train_sentences, train_tags)

# Prepare data tensors
input_ids = torch.tensor([item['input_ids'] for item in tokenized_data], dtype=torch.long)
labels = torch.tensor([item['labels'] for item in tokenized_data], dtype=torch.long)

train_data = TensorDataset(input_ids, labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

model.save_pretrained('pos_model')
model.eval()

original_evalset = pd.read_csv("dev-ur-romanized.csv")

eval_sentences = original_evalset['Sentence'].tolist()
eval_romanizations = original_evalset['Romanization'].tolist()
eval_tags = original_evalset['Tags'].tolist()
eval_sentences, eval_tags = convert_with_romanization(eval_sentences, eval_romanizations, eval_tags)
tokenized_data = format_with_romanization(eval_sentences, eval_tags)

from sklearn.metrics import f1_score

def evaluate_f1(tokenized_data, eval_sentences):
    y_pred, y_true = [], []

    for i in range(len(tokenized_data)):
        text = eval_sentences[i]
        inputs = tokenizer(text, return_tensors='pt')
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        predicted_labels = outputs.logits.argmax(dim=-1).tolist()[0]
        token_labels = tokenized_data[i]['labels']
        token_importance = tokenized_data[i]['importance']
        is_start = tokenized_data[i]['is_start']

        # Extract the first token of each word using importance and its preceding token
        for j, importance_flag in enumerate(token_importance):
            if importance_flag == 1 and is_start[j]:  # First token logic
                y_true.append(token_labels[j])
                y_pred.append(predicted_labels[j])

    return y_true, y_pred

# Evaluate model
y_true, y_pred = evaluate_f1(tokenized_data, eval_sentences)

# Calculate and print F1 score
score = f1_score(y_true, y_pred, average='macro')
print(f'Word-level F1 score using first token: {score}')