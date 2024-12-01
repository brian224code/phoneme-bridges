import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from transformers import BertTokenizerFast, BertForTokenClassification


entity_types = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

num_labels = len(entity_types)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', device_map=device)


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
    return sentences, tags


def format(sentences, tags):
    tokenized_data = []
    for i in range(len(sentences)):
        text = sentences[i]
        entities = tags[i]
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
        labels = ['O'] * len(tokens)
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
        label_ids += [entity_types.index('O')] * padding_length
        is_start += [False] * padding_length
        tokenized_data.append({
            'input_ids': [input_ids[j] for j in range(len(input_ids))],
            'labels': [label_ids[j] for j in range(len(label_ids))],
            'is_start': [is_start[j] for j in range(len(input_ids))]
        })
    return tokenized_data


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

        labels = ['O'] * len(input_ids)
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


def process_non_romanized(filename):
    evalset = pd.read_csv(filename)
    sentences = evalset['Sentence'].tolist()
    tags = evalset['Tags'].tolist()
    sentences, tags = convert(sentences, tags)
    tokenized_data = format(sentences, tags)
    return tokenized_data, sentences


def process_romanized(filename):
    evalset = pd.read_csv(filename)
    sentences = evalset['Sentence'].tolist()
    romanizations = evalset['Romanization'].tolist()
    tags = evalset['Tags'].tolist()
    sentences, tags = convert_with_romanization(sentences, romanizations, tags)
    tokenized_data = format_with_romanization(sentences, tags)
    return tokenized_data, sentences


def evaluate_non_romanized(model, tokenized_data, sentences):
    y_pred, y_true = [], []

    for i in range(len(tokenized_data)):
        text = sentences[i]
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


def evaluate_romanized(model, tokenized_data, sentences):
    y_pred, y_true = [], []

    for i in range(len(tokenized_data)):
        text = sentences[i]
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


def paired_bootstrap_resampling(
    model_1,
    model_2,
    tokenized_data_1,
    tokenized_data_2,
    sentences_1,
    sentences_2,
    n_iterations=10000,
    n_samples=1000,
    alpha=0.05
):
    indices = np.arange(len(tokenized_data_1))
    performance_diffs = []

    for _ in range(n_iterations):
        # Sample indices with replacement to create a bootstrap sample
        sample_indices = np.random.choice(indices, size=n_samples, replace=True)

        # Extract the samples for the current iteration
        sample_tokenized_data_1 = [tokenized_data_1[i] for i in sample_indices]
        sample_tokenized_data_2 = [tokenized_data_2[i] for i in sample_indices]
        sample_sentences_1 = [sentences_1[i] for i in sample_indices]
        sample_sentences_2 = [sentences_2[i] for i in sample_indices]

        # Calculate the performance metric for both models
        y_true_1, y_pred_1 = evaluate_non_romanized(model_1, sample_tokenized_data_1, sample_sentences_1)
        y_true_2, y_pred_2 = evaluate_romanized(model_2, sample_tokenized_data_2, sample_sentences_2)
        score_1 = f1_score(y_true_1, y_pred_1, average='macro')
        score_2 = f1_score(y_true_2, y_pred_2, average='macro')

        # Compute the difference in performance
        performance_diffs.append(score_2 - score_1)

    performance_diffs = np.array(performance_diffs)
    mean_diff = np.mean(performance_diffs)
    lower_bound = np.percentile(performance_diffs, 100 * (alpha / 2))
    upper_bound = np.percentile(performance_diffs, 100 * (1 - alpha / 2))
    conf_interval = (lower_bound, upper_bound)

    # Calculate the p-value for the observed difference
    p_value = np.mean(performance_diffs <= 0) if mean_diff > 0 else np.mean(performance_diffs >= 0)
    p_value *= 2  # Two-tailed test

    return mean_diff, conf_interval, p_value


# Example usage
if __name__ == "__main__":
    model_1 = BertForTokenClassification.from_pretrained('ner_model', num_labels=num_labels, device_map=device)
    model_2 = BertForTokenClassification.from_pretrained('ner_model_romanized', num_labels=num_labels, device_map=device)

    tokenized_data_1, sentences_1 = process_non_romanized('dev-ur.csv')
    tokenized_data_2, sentences_2 = process_romanized('dev-ur-romanized.csv')

    mean_diff_f1, conf_int_f1, p_value_f1 = paired_bootstrap_resampling(
        model_1,
        model_2,
        tokenized_data_1,
        tokenized_data_2,
        sentences_1,
        sentences_2,
        n_iterations=1000,
        n_samples=1000,
        alpha=0.05
    )

    print("\nMacro-F1 Score Comparison:")
    print(f"Mean Difference: {mean_diff_f1:.4f}")
    print(f"95% Confidence Interval: [{conf_int_f1[0]:.4f}, {conf_int_f1[1]:.4f}]")
    print(f"P-value: {p_value_f1:.4f}")
