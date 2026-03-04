import os
import json
from bs4 import BeautifulSoup

INPUT_DIR = "/home/swaminathanj/LIV/content/dataset/Project_CodeNet/problem_descriptions"
OUTPUT_FILE = "problem_descriptions.jsonl"

def extract_problem_description(html_path):
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "lxml")

    text_blocks = []

    # Strategy 1: <h2>Problem</h2> or <h3>Problem Statement</h3>
    problem_headers = soup.find_all(
        lambda tag: tag.name in ["h2", "h3"] and
        tag.get_text(strip=True).lower() in ["problem", "problem statement"]
    )

    if problem_headers:
        header = problem_headers[0]
        for sibling in header.find_next_siblings():
            if sibling.name in ["h2", "h3"] and sibling.get_text(strip=True).lower() in [
                "input", "constraints", "output"
            ]:
                break
            if sibling.name in ["p", "div", "section", "ul", "pre"]:
                text_blocks.append(sibling.get_text(" ", strip=True))

    # Strategy 2: fallback – collect <p> after <h1> until Input
    if not text_blocks:
        h1 = soup.find("h1")
        if h1:
            for sibling in h1.find_next_siblings():
                if sibling.name in ["h2", "h3"] and sibling.get_text(strip=True).lower() == "input":
                    break
                if sibling.name in ["p"]:
                    text_blocks.append(sibling.get_text(" ", strip=True))

    description = " ".join(text_blocks)
    description = " ".join(description.split())  # normalize whitespace

    return description if description else None


def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for fname in sorted(os.listdir(INPUT_DIR)):
            if not fname.endswith(".html"):
                continue

            problem_id = os.path.splitext(fname)[0]  # e.g., p02120
            html_path = os.path.join(INPUT_DIR, fname)

            description = extract_problem_description(html_path)
            if description is None:
                continue

            record = {
                "problem_id": problem_id,
                "description": description
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved problem descriptions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
