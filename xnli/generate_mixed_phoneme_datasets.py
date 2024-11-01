import pandas as pd
import random
import matplotlib.pyplot as plt
import statistics
import math


def read_file(file_path, probability):
    # Read the data using pandas
    data = pd.read_csv(
        file_path,
        engine='python'
    )

    # Initialize list to hold mixed sentences
    mixed_phoneme_sentences = [[],[]]

    # exchange_data = []

    # Iterate over the DataFrame
    for index, row in data.iterrows():
        for sentence_number in [0, 1]:
            mixed_phoneme_sentence = []

            sentence = row['Sentence'+str(sentence_number+1)].split()
            phonemes = row['Romanization'+str(sentence_number+1)].split()

            total_words = len(sentence)
            num_phonemes_to_exchange = math.ceil(total_words * probability)
            indices_to_exchange = random.sample(range(total_words), num_phonemes_to_exchange)

            for i in range(total_words):
                if i in indices_to_exchange:
                    mixed_phoneme_sentence.append(phonemes[i])
                else:
                    mixed_phoneme_sentence.append(sentence[i])

            # exchange_data.append(float(num_phonemes_to_exchange) / total_words)

            mixed_phoneme_sentence = ' '.join(mixed_phoneme_sentence)

            mixed_phoneme_sentences[sentence_number].append(mixed_phoneme_sentence)

    # Display exchange stats 
    # print('Mean: ', statistics.mean(exchange_data))
    # print('Median: ', statistics.median(exchange_data))
    # print('Stdev: ', statistics.stdev(exchange_data))
    # plt.hist(exchange_data)
    # plt.show()

    # Add new columns to dataframe
    data['Mixed1'] = mixed_phoneme_sentences[0]
    data['Mixed2'] = mixed_phoneme_sentences[1]

    # Save to original csv
    data.to_csv(file_path, index=False)

    # Display the first few sentences
    # print(data.head())

def process_datasets():

    # Probability of each word to be changed to phoneme ie percent of dataset to be changed to phoneme
    probability = 0.25

    # List of datasets
    datasets = [
        '../xnli/train-hi-romanized.csv',
        '../xnli/dev-hi-romanized.csv',
        '../xnli/train-ur-romanized.csv', 
        '../xnli/dev-ur-romanized.csv'
    ]

    # Process each dataset
    for file in datasets:
        print(f"Processing {file}...")
        read_file(file, probability)
        print(f"Saved to {file}")

if __name__ == "__main__":
    process_datasets()
