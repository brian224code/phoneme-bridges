import pandas as pd
import csv

def read_file(file_path, output_file):
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

    # Initialize lists to hold sentences and tags
    sentences = []
    tags = []

    # Temporary lists to hold tokens and tags for the current sentence
    temp_sentence = []
    temp_tags = []

    # Iterate over the DataFrame
    for index, row in data.iterrows():
        if pd.isnull(row['Token']):
            # End of a sentence
            if temp_sentence:
                sentences.append(temp_sentence)
                tags.append(temp_tags)
                temp_sentence = []
                temp_tags = []
        else:
            # Add token and tag to the temporary lists
            temp_sentence.append(row['Token'])
            temp_tags.append(row['Tag'])

    # Add the last sentence if it wasn't added
    if temp_sentence:
        sentences.append(temp_sentence)
        tags.append(temp_tags)

    # Create a DataFrame with sentences and their corresponding tags
    df_sentences = pd.DataFrame({'Sentence': sentences, 'Tags': tags})

    # Save the DataFrame as a CSV file
    df_sentences.to_csv(output_file, index=False)

    # Display the first few sentences
    print(df_sentences.head())

def process_datasets():
    # List of datasets
    datasets = [
        ('../panx/train-hi.tsv', '../panx/train-hi.csv'),
        ('../panx/dev-hi.tsv', '../panx/dev-hi.csv'),
        ('../panx/train-ur.tsv', '../panx/train-ur.csv'),
        ('../panx/dev-ur.tsv', '../panx/dev-ur.csv')
    ]

    # Process each dataset
    for input_file, output_file in datasets:
        print(f"Processing {input_file}...")
        read_file(input_file, output_file)
        print(f"Saved to {output_file}")

if __name__ == "__main__":
    process_datasets()