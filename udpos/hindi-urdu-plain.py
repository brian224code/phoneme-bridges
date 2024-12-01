import pandas as pd
original_dataset = pd.read_csv('train-hi.csv')

train_sentences = original_dataset['Sentence'].tolist()
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

from transformers import BertTokenizer, BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch

entity_types = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
num_labels = len(entity_types)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', device_map=device)
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
        is_start = [False] * len(tokens)
        for start, end, entity_type in entities:
            prefix_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text[:start])))[:-1]
            start_token = len(prefix_tokens)
            entity_tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text[start:end])))[1:-1]
            end_token = start_token + len(entity_tokens)
            word_start = True
            for j in range(start_token, end_token):
                labels[j] = entity_type
                is_start[j] = word_start
                word_start = False
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = [entity_types.index(label) for label in labels]
        padding_length = tokenizer.model_max_length - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * padding_length
        label_ids += [entity_types.index('ADJ')] * padding_length
        is_start += [False] * padding_length
        tokenized_data.append({
            'input_ids': [input_ids[j] for j in range(len(input_ids))],
            'labels': [label_ids[j] for j in range(len(label_ids))],
            'is_start': [is_start[j] for j in range(len(input_ids))]
        })
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

original_dataset = pd.read_csv('train-ur.csv')

train_sentences = original_dataset['Sentence'].tolist()
train_tags = original_dataset['Tags'].tolist()

convert(train_sentences, train_tags)

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

model.save_pretrained('pos_model')

original_evalset = pd.read_csv("dev-ur.csv")

eval_sentences = original_evalset['Sentence'].tolist()
eval_tags = original_evalset['Tags'].tolist()
convert(eval_sentences, eval_tags)
tokenized_data = format(eval_sentences, eval_tags)

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
        is_start = tokenized_data[i]['is_start']

        # Extract the first token of each word using importance and its preceding token
        for j in range(len(token_labels)):
            if is_start[j]:  # First token logic
                y_true.append(token_labels[j])
                y_pred.append(predicted_labels[j])

    return y_true, y_pred

# Evaluate model
y_true, y_pred = evaluate_f1(tokenized_data, eval_sentences)

# Calculate and print F1 score
score = f1_score(y_true, y_pred, average='macro')
print(f'Word-level F1 score using first token: {score}')