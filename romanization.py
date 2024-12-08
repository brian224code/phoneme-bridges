from ai4bharat.transliteration import XlitEngine
import pandas as pd

def obtainRomanizations(language, input_file, output_file):
    e = XlitEngine(src_script_type="indic", beam_width=10, rescore=False)

    df = pd.read_csv(input_file)
    print(df.head())

    df.insert(2, "Romanization", [e.translit_sentence(df['Sentence'][i], lang_code=language) for i in range(len(df['Sentence']))], True)
    print(df.head())

    # Save the resulting DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Saved the DataFrame to {output_file}")

if __name__ == "__main__":
    # Example usage
    obtainRomanizations('hi', 'dev-hi.csv', 'dev-hi-romanized.csv')