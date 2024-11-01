import json
import re

def generate_dict():
    cleaned_dict = {}

    with open("../kaikki.org-dictionary-Urdu.jsonl", encoding="utf-8") as f:
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


def generate_romanized_dict():
    romanization_dict = {}

    with open("../kaikki.org-dictionary-Urdu.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            
            if "forms" in data:
                for form in data['forms']:
                    try:
                        if 'tags' not in form:
                            romanization_dict[form['form']] = form['roman']
                    except:
                        continue

                    try:
                        if 'romanization' in form['tags']:
                            romanization_dict[data['word']] = form['form']
                        elif 'Hindi' not in form['tags']:
                            romanization_dict[form['form']] = form['roman']
                    except:
                        continue

    cleaned_romanization_dict = {}
    for word, romanization in romanization_dict.items():
        if 'span' in word:
            continue
        split_words = re.split('، |،| ', word)
        split_romanizations = re.split(', |,| |-', romanization)

        if len(split_romanizations) == len(split_words):
            for split_word, split_romanization in zip(split_words, split_romanizations):
                cleaned_romanization_dict[split_word] = split_romanization

        cleaned_romanization_dict[word] = romanization

    print('len dict: ', len(cleaned_romanization_dict))


    # Check hits on missing words with romanization dict

    # count = 0                    
    # with open("../missing_urdu_words.txt", encoding="utf-8") as f:
    #     for line in f:
    #         line = line.rstrip()
    #         split_line =  re.split('، |،| ', line)

    #         for word in split_line:
    #             if word not in cleaned_romanization_dict:
    #                 if line in cleaned_romanization_dict:
    #                     print(line)
    #                 break
    #         else:
    #             count += 1

    # print('Number of hits: ', count)

            
    
if __name__ == '__main__':
    # remove_slashes()
    generate_romanized_dict()