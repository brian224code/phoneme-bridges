import pandas as pd
import csv
import os

def read_file(file_path, output_file):
    # Read the data using pandas
    data = pd.read_csv(
        file_path,
        sep='\t',
        header=None,
        names=['Sentence1', 'Sentence2', 'Label']
    )

    # Create a DataFrame with the required columns
    df_sentences = pd.DataFrame({
        'Sentence1': data['Sentence1'].str.replace('\t', ' '),
        'Sentence2': data['Sentence2'].str.replace('\t', ' '),
        'Label': data['Label']
    })

    # Save the DataFrame as a CSV file
    df_sentences.to_csv(output_file, index=False)

    # Display the first few sentences
    print(df_sentences.head())

def process_datasets():
    # List of datasets
    datasets = [
        ('../../xnli/dev-hi.tsv', '../../xnli/dev-hi.csv'),
        ('../../xnli/dev-ur.tsv', '../../xnli/dev-ur.csv')
    ]

    # Process each dataset
    for input_file, output_file in datasets:
        print(f"Processing {input_file}...")
        read_file(input_file, output_file)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    process_datasets()