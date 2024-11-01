import pandas as pd
from sklearn.model_selection import train_test_split

langs = ['ur', 'hi']

for lang in langs: 
    # Read the CSV file
    df = pd.read_csv(f'../../xnli/dev-{lang}-romanized.csv')
    print(f"Original DataFrame shape: {df.shape}")

    # Split the DataFrame into train and dev sets (80%-20% split)
    train_df, dev_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train DataFrame shape: {train_df.shape}")
    print(f"Dev DataFrame shape: {dev_df.shape}")

    # Save the train and dev sets to new CSV files
    train_file = f'../../xnli/train-{lang}-romanized.csv'
    dev_file = f'../../xnli/dev-{lang}-romanized-split.csv'

    train_df.to_csv(train_file, index=False)
    dev_df.to_csv(dev_file, index=False)

    print(f"Saved the train set to {train_file}")
    print(f"Saved the dev set to {dev_file}")
