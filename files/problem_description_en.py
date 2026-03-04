### Run on Colab
# All of the Project CodeNet problem descriptions are in one of the two languages: English or Japaneese.
# This code converts japanese problem descriptions to english.


!pip install googletrans==4.0.0-rc1 langdetect

import json
from langdetect import detect
from googletrans import Translator

translator = Translator()

input_file = "/content/problem_descriptions.jsonl"
output_file = "/content/problem_description_en.jsonl"

def is_japanese(text):
    try:
        return detect(text) == "ja"
    except:
        return False

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        item = json.loads(line)
        desc = item["description"]

        if is_japanese(desc):
            translated = translator.translate(desc, src="ja", dest="en").text
            item["description"] = translated

        fout.write(json.dumps(item, ensure_ascii=False) + "\n")

import json
from langdetect import detect

ORIGINAL_FILE = "/content/problem_descriptions.jsonl"
TRANSLATED_FILE = "/content/problem_description_en.jsonl"
MAX_EXAMPLES = 3  # change as needed


def is_japanese(text):
    try:
        return detect(text) == "ja"
    except:
        return False


with open(ORIGINAL_FILE, "r", encoding="utf-8") as f_orig, \
     open(TRANSLATED_FILE, "r", encoding="utf-8") as f_trans:

    count = 0

    for orig_line, trans_line in zip(f_orig, f_trans):
        orig_item = json.loads(orig_line)
        trans_item = json.loads(trans_line)

        orig_desc = orig_item["description"]
        trans_desc = trans_item["description"]

        if is_japanese(orig_desc):
            print("=" * 80)
            print(f"Problem ID : {orig_item['problem_id']}")
            print("\n[Original – Japanese]")
            print(orig_desc)
            print("\n[Translated – English]")
            print(trans_desc)
            print("=" * 80)
            print()

            count += 1
            if count >= MAX_EXAMPLES:
                break