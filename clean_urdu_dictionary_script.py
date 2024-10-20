import json

cleaned_dict = {}

def generate_dict():
    with open("kaikki.org-dictionary-Urdu.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            if "sounds" in data:
                if "ipa" in data["sounds"][0]:
                    morpheme = data["word"]
                    phoneme = data["sounds"][0]["ipa"]
                    cleaned_dict[morpheme] = phoneme

    with open("urdu_grapheme_to_phoneme.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_dict, f, ensure_ascii=False, indent=4)

def remove_slashes():
    # Read the JSON file
    with open("../urdu_grapheme_to_phoneme.json", "r", encoding="utf-8") as file:
        urdu_dict = json.load(file)

    # Function to remove surrounding slashes
    def remove_slashes(pronunciation):
        if pronunciation.startswith('/') and pronunciation.endswith('/'):
            return pronunciation[1:-1]
        return pronunciation

    # Update the dictionary by removing slashes from pronunciations
    updated_dict = {word: remove_slashes(pronunciation) for word, pronunciation in urdu_dict.items()}

    # Write the updated dictionary back to the JSON file
    with open("urdu_grapheme_to_phoneme.json", "w", encoding="utf-8") as file:
        json.dump(updated_dict, file, ensure_ascii=False, indent=4)

    print("Processing complete. Slashes have been removed from IPA pronunciations.")

    # Optional: Print a few examples to verify
    print("\nExample entries from the updated dictionary:")
    for word, pronunciation in list(updated_dict.items())[:5]:  # Print first 5 entries
        print(f"{word}: {pronunciation}")
    
if __name__ == '__main__':
    remove_slashes()