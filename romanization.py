from ai4bharat.transliteration import XlitEngine
import pandas as pd
import random
import math

def obtainRomanizations(language, input_file, output_file):
    e = XlitEngine(src_script_type="indic", beam_width=10, rescore=False)

    df = pd.read_csv(input_file)
    print(df.head())

    df.insert(2, "Romanization", [e.translit_sentence(df['Sentence'][i], lang_code=language) for i in range(len(df['Sentence']))], True)
    print(df.head())

    # Save the resulting DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Saved the DataFrame to {output_file}")

def randomReplacement(file_path, probability):
    # Read the data using pandas
    data = pd.read_csv(
        file_path,
        engine='python'
    )

    # Initialize list to hold mixed sentences
    mixed_phoneme_sentences = []

    # Iterate over the DataFrame
    for index, row in data.iterrows():
        mixed_phoneme_sentence = []

        sentence = row['Sentence'].split()
        phonemes = row['Romanization'].split()

        total_words = len(sentence)
        num_phonemes_to_exchange = math.ceil(total_words * probability)
        indices_to_exchange = random.sample(range(total_words), num_phonemes_to_exchange)

        for i in range(total_words):
            if i in indices_to_exchange:
                mixed_phoneme_sentence.append(phonemes[i])
            else:
                mixed_phoneme_sentence.append(sentence[i])

        mixed_phoneme_sentence = ' '.join(mixed_phoneme_sentence)

        mixed_phoneme_sentences.append(mixed_phoneme_sentence)

    # Add new column to dataframe
    data['Mixed'] = mixed_phoneme_sentences

    # Save to original csv
    data.to_csv(file_path, index=False)

def process_datasets(datasets, probability):
    # Process each dataset
    for file in datasets:
        print(f"Processing {file}...")
        randomReplacement(file, probability)
        print(f"Saved to {file}")

if __name__ == "__main__":
    # Example usage to generate romanizations
    obtainRomanizations('hi', 'dev-hi.csv', 'dev-hi-romanized.csv')

    # Example usage to randomly replace 25% of words in dataset with romanizations
    # probability = 0.25
    # datasets = [
    #     '../udpos/train-hi-romanized.csv',
    #     '../udpos/dev-hi-romanized.csv',
    #     '../udpos/train-ur-romanized.csv', 
    #     '../udpos/dev-ur-romanized.csv'
    # ]
    # process_datasets(datasets, probability)