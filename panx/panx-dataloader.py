import pandas as pd
import csv
import os
import epitran
import json

# Load the Urdu dictionary
with open("../../urdu_grapheme_to_phoneme.json", encoding="utf-8") as f:
    urdu_dict = json.load(f)

# Create Epitran instances for Hindi and Urdu
epi_hi = epitran.Epitran('hin-Deva')
epi_ur = epitran.Epitran('urd-Arab')

def join_tokens(tokens):
    # Join tokens with spaces
    return " ".join(tokens)

def transliterate_word(word, language):
    unknown_token = "<unk>"
    try:
        if language == 'hi':
            return epi_hi.transliterate(word)
        elif language == 'ur':
            return urdu_dict.get(word, unknown_token)
    except Exception as e:
        print(f"Error during {language} transliteration: {e}")
        return unknown_token

def transliterate_sentence(sentence, language):
    transliterated_words = [transliterate_word(word, language) for word in sentence]
    return " ".join(transliterated_words)

def read_file(file_path, output_file, language, missing_words):
    # Read the data using pandas
    data = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        names=['Token', 'Tag'],
        engine='python',
        quoting=csv.QUOTE_NONE,
        skip_blank_lines=False
    )

    # Initialize lists to hold sentences, tags, and IPA transliterations
    sentences = []
    tags = []
    ipas = []

    # Temporary lists to hold tokens and tags for the current sentence
    temp_sentence = []
    temp_tags = []

    # Iterate over the DataFrame
    for index, row in data.iterrows():
        if pd.isnull(row['Token']):
            # End of a sentence
            if temp_sentence:
                sentences.append(join_tokens(temp_sentence))
                tags.append(" ".join(temp_tags))
                ipas.append(transliterate_sentence(temp_sentence, language))
                temp_sentence = []
                temp_tags = []
        else:
            # Add token and tag to the temporary lists
            temp_sentence.append(row['Token'])
            temp_tags.append(row['Tag'])

            # Track missing Urdu words
            if language == 'ur' and row['Token'] not in urdu_dict:
                missing_words.add(row['Token'])

    # Add the last sentence if it wasn't added
    if temp_sentence:
        sentences.append(join_tokens(temp_sentence))
        tags.append(" ".join(temp_tags))
        ipas.append(transliterate_sentence(temp_sentence, language))

    # Create a DataFrame with sentences, tags, and their corresponding IPA transliterations
    df_sentences = pd.DataFrame({'Sentence': sentences, 'Tags': tags, 'IPA': ipas})

    # Save the DataFrame as a CSV file
    df_sentences.to_csv(output_file, index=False)

    # Display the first few sentences
    print(df_sentences.head())
def process_datasets():
    # List of datasets
    datasets = [
        ('../../panx/train-hi.tsv', '../../panx/train-hi.csv', 'hi'),
        ('../../panx/dev-hi.tsv', '../../panx/dev-hi.csv', 'hi'),
        ('../../panx/train-ur.tsv', '../../panx/train-ur.csv', 'ur'),
        ('../../panx/dev-ur.tsv', '../../panx/dev-ur.csv', 'ur')
    ]

    # Counters for Urdu words
    total_urdu_words = 0
    transliterated_urdu_words = 0

    # Set to track missing Urdu words
    missing_words = set()

    # Process each dataset
    for input_file, output_file, language in datasets:
        print(f"Processing {input_file}...")
        if language == 'ur':
            with open(input_file, encoding="utf-8") as f:
                for line in f:
                    word = line.split('\t')[0]
                    if word:  # only count non-empty words
                        total_urdu_words += 1
                        if word in urdu_dict:
                            transliterated_urdu_words += 1
        read_file(input_file, output_file, language, missing_words)
        print(f"Saved to {output_file}")

    if total_urdu_words > 0:
        percentage_transliterated = (transliterated_urdu_words / total_urdu_words) * 100
        print(f"Percentage of Urdu words successfully transliterated: {percentage_transliterated:.2f}%")

    # Save missing Urdu words to a separate file
    with open("missing_urdu_words.txt", "w", encoding="utf-8") as file:
        for word in sorted(missing_words):
            file.write(word + "\n")
    print(f"Missing Urdu words saved to missing_urdu_words.txt")

if __name__ == "__main__":
    process_datasets()