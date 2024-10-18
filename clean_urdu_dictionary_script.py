import json

cleaned_dict = {}

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