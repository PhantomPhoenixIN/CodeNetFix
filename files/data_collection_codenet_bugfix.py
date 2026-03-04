#!/usr/bin/env python3
"""
codenet_bugfix_extractor_alllangs.py

Extract bug->fix pairs from Project CodeNet for all languages (no explicit language list).

Output JSONL schema (one JSON object per line):
{
  "problem_id": "...",
  "user_id": "...",
  "language": "<folder name>",
  "buggy_submission_id": "...",
  "fixed_submission_id": "...",
  "status_buggy": "...",
  "status_fixed": "Accepted",
  "buggy_code": "...",
  "fixed_code": "...",
  "test_cases": {...}
}

Notes:
- For each problem, every language folder is processed.
- Metadata rows are matched to a folder language if their normalized forms match (flexible).
- If a metadata CSV lacks 'user_id' the script skips that language for that problem.
- Outputs a summary table and two CSVs: language distribution & top users.
"""

import os
import re
import json
import time
from collections import Counter, defaultdict, OrderedDict
from typing import Optional, Dict, List

import pandas as pd
from bs4 import BeautifulSoup

# ----------------------------
# CONFIG - update these paths if needed
# ----------------------------
DATA_DIR = "/home/swaminathanj/LIV/content/dataset/Project_CodeNet/data"
METADATA_DIR = "/home/swaminathanj/LIV/content/dataset/Project_CodeNet/metadata"
HTML_DIR = "/home/swaminathanj/LIV/content/dataset/Project_CodeNet/problem_descriptions"
OUT_FILE = "/home/swaminathanj/CodeFixNet/content/datasets/final_codenet_bugfix_pairs_alllangs.jsonl"

# CSV summary outputs
OUT_DIR = os.path.dirname(OUT_FILE)
LANGUAGE_DIST_CSV = os.path.join(OUT_DIR, "language_distribution.csv")
TOP_USERS_CSV = os.path.join(OUT_DIR, "top_users.csv")

MAX_PAIRS_PER_USER_PER_PROBLEM = None  # keep all earlier non-accepted by a user
MIN_CODE_CHARS = 10
FLUSH_EVERY = 100
VERBOSE = True

# Quick dry-run option for testing smaller subset
DRY_RUN = False
DRY_RUN_LIMIT = 200  # number of problems to process in dry-run
# ----------------------------


def normalize_lang_str(s: Optional[str]) -> str:
    """Normalize language strings for flexible matching (case-insensitive, common mappings)."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("c++", "cpp")
    s = s.replace("c#", "csharp")
    s = s.replace("f#", "fsharp")
    # remove spaces and non-alphanumerics except hyphen
    s = re.sub(r'[^a-z0-9\-]', '', s)
    return s


def looks_like_code(s: str) -> bool:
    """Heuristic check: does a string look like source code?"""
    if not s:
        return False
    low = s.lower()
    code_signals = [
        "#include", "int main", "printf(", "cout<<", "std::", "using namespace",
        "class ", "def ", "public static", "import java", "system.out", "package ",
        "using std", "console.writeline", "function(", "<?php"
    ]
    if any(sig in low for sig in code_signals):
        return True
    if s.count(";") >= 2 or (s.count("{") + s.count("}")) >= 2:
        return True
    return False


# ----------------------------
# Strict test-case extractor
# ----------------------------
def extract_test_cases_strict(html_file, max_lines_threshold=1000):
    """
    Strict extractor for Sample Input / Output formats.
    Only recognizes:
      - "Sample Input" / "Sample Input N"
      - "Output for the Sample Input" / "Output for the Sample Input N"
      - "Sample Output" / "Sample Output N"
    Returns OrderedDict of {"TestCase_1": {"input": "...", "output": "...", "suspicious": bool}, ...}
    """
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception:
        return OrderedDict()

    input_re = re.compile(r'^\s*sample\s+input(?:\s*(\d+))?\s*$', flags=re.I)
    output_re1 = re.compile(r'^\s*output\s+for\s+the\s+sample\s+input(?:\s*(\d+))?\s*$', flags=re.I)
    output_re2 = re.compile(r'^\s*sample\s+output(?:\s*(\d+))?\s*$', flags=re.I)

    heading_tags = ["h1", "h2", "h3", "h4", "strong", "b", "p", "div"]
    nodes = []
    for tag in soup.find_all(heading_tags + ["pre"]):
        nodes.append(tag)

    # Build list of labeled headings (only the recognized Sample Input/Output headings) and keep their positions
    labeled = []
    for n in nodes:
        if n.name == "pre":
            labeled.append((n, "pre", None))
            continue
        text = (n.get_text(" ", strip=True) or "").strip()
        if not text:
            continue
        m_in = input_re.match(text)
        if m_in:
            num = int(m_in.group(1)) if m_in.group(1) else None
            labeled.append((n, "input", num))
            continue
        m_out1 = output_re1.match(text)
        if m_out1:
            num = int(m_out1.group(1)) if m_out1.group(1) else None
            labeled.append((n, "output", num))
            continue
        m_out2 = output_re2.match(text)
        if m_out2:
            num = int(m_out2.group(1)) if m_out2.group(1) else None
            labeled.append((n, "output", num))
            continue
        # ignore any other headings (including plain "Input"/"Output")

    # Helper to find next pre after a node by index in labeled list
    def next_pre_after_node(start_idx):
        for j in range(start_idx + 1, len(labeled)):
            node_j, typ_j, num_j = labeled[j]
            if typ_j == "pre":
                return node_j, j
        return None, None

    test_cases = OrderedDict()
    case_idx = 1

    # Walk through labeled list; only start from items labeled "input"
    for i, (node, typ, num) in enumerate(labeled):
        if typ != "input":
            continue

        # 1) Find the next <pre> after this input heading — candidate input_pre
        input_pre, pre_idx = next_pre_after_node(i)
        valid_input_pre = False
        if input_pre is not None:
            intervening_heading_found = False
            for k in range(i + 1, pre_idx):
                if labeled[k][1] in ("input", "output"):
                    intervening_heading_found = True
                    break
            if not intervening_heading_found:
                valid_input_pre = True

        if not valid_input_pre:
            # no valid pre (or another labeled heading intervened) -> create a suspicious empty testcase
            test_cases[f"TestCase_{case_idx}"] = {"input": "", "output": "", "suspicious": True}
            case_idx += 1
            continue

        input_text = input_pre.get_text(strip=True)

        # 2) Scan forward from input_pre to find the matching output heading with same number (num)
        output_text = ""
        found_output = False
        j = pre_idx + 1
        while j < len(labeled):
            node_j, typ_j, num_j = labeled[j]
            if typ_j == "output":
                if (num is None and num_j is None) or (num is not None and num_j == num):
                    # next pre after this output heading is the actual output pre
                    out_pre_node = None
                    for k in range(j + 1, len(labeled)):
                        if labeled[k][1] == "pre":
                            out_pre_node = labeled[k][0]
                            break
                    if out_pre_node:
                        output_text = out_pre_node.get_text(strip=True)
                    else:
                        output_text = ""
                    found_output = True
                    break
            elif typ_j == "input":
                # another input appeared before any output -> treat as missing output
                break
            j += 1

        suspicious = False
        if not found_output:
            suspicious = True
        if len(input_text.splitlines()) > max_lines_threshold or len(output_text.splitlines()) > max_lines_threshold:
            suspicious = True

        test_cases[f"TestCase_{case_idx}"] = {"input": input_text, "output": output_text, "suspicious": suspicious}
        case_idx += 1

    return test_cases


# alias for compatibility if any call uses extract_test_cases(...)
extract_test_cases = extract_test_cases_strict


# ----------------------------
# Read submission file robustly
# ----------------------------
# cache of directory listings to avoid repeated os.listdir
_dir_listing_cache: Dict[str, List[str]] = {}


def read_submission_file(lang_dir: str, submission_id: str, filename_ext: str) -> Optional[str]:
    """Read submission file robustly with a few fallback patterns."""
    if not os.path.isdir(lang_dir):
        return None
    ext = (filename_ext or "").lstrip(".")
    candidates = []
    if ext:
        candidates.append(f"{submission_id}.{ext}")
    candidates.append(str(submission_id))

    # first quick direct attempts
    for name in candidates:
        p = os.path.join(lang_dir, name)
        if os.path.exists(p) and os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    return f.read().strip()
            except Exception:
                return None

    # fallback: search files in the directory but cache listing
    listing = _dir_listing_cache.get(lang_dir)
    if listing is None:
        try:
            listing = os.listdir(lang_dir)
        except Exception:
            listing = []
        _dir_listing_cache[lang_dir] = listing

    for fname in listing:
        if fname.startswith(str(submission_id)):
            p = os.path.join(lang_dir, fname)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8", errors="replace") as f:
                        return f.read().strip()
                except Exception:
                    continue
    return None


def normalize_code(code: str) -> str:
    """Simple normalization used for exact dedupe within a user's buggy submissions."""
    if not code:
        return ""
    s = code.strip()
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s


def get_time_or_id(row: pd.Series) -> Optional[float]:
    """Ordering key for submissions (prefer submission_time, fallback to submission_id)."""
    if 'submission_time' in row and pd.notna(row['submission_time']):
        try:
            return float(row['submission_time'])
        except Exception:
            pass
    if 'submission_id' in row and pd.notna(row['submission_id']):
        try:
            return float(row['submission_id'])
        except Exception:
            # fallback: try numeric portion
            try:
                return float(re.sub(r'\D', '', str(row['submission_id'])))
            except Exception:
                pass
    return None


def parse_metadata(path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return None


def extract_pairs_for_problem(pid: str, p_dir: str, meta_df: pd.DataFrame, html_dir: str, diag: dict) -> List[dict]:
    """Process every language folder under p_dir and return found bug->fix pairs."""
    pairs = []
    folder_langs = [d for d in sorted(os.listdir(p_dir)) if os.path.isdir(os.path.join(p_dir, d))]
    html_file = os.path.join(html_dir, f"{pid}.html")
    if os.path.exists(html_file):
        test_cases_for_problem = extract_test_cases_strict(html_file)
    else:
        test_cases_for_problem = OrderedDict()

    # Precompute normalized metadata language mapping: normalized -> list of actual meta-language strings
    meta_has_language = 'language' in meta_df.columns
    norm_meta_map = defaultdict(list)
    if meta_has_language:
        for v in meta_df['language'].dropna().astype(str).unique().tolist():
            norm_meta_map[normalize_lang_str(v)].append(v)

    for folder_lang in folder_langs:
        lang_dir = os.path.join(p_dir, folder_lang)

        # Determine which metadata rows correspond to this folder_lang.
        matched_meta_rows = []
        if meta_has_language:
            exact_matches = meta_df[meta_df['language'].astype(str).str.lower() == folder_lang.lower()]
            if not exact_matches.empty:
                matched_meta_rows = exact_matches.to_dict('records')
            else:
                n_folder = normalize_lang_str(folder_lang)
                meta_vals = norm_meta_map.get(n_folder, [])
                if meta_vals:
                    matched_meta_rows = []
                    for mv in meta_vals:
                        matched_meta_rows.extend(meta_df[meta_df['language'] == mv].to_dict('records'))
                else:
                    matched_meta_rows = []

        if not matched_meta_rows:
            # permissive fallback: if metadata has user_id and submission_id, use all rows for this problem
            if 'user_id' in meta_df.columns and 'submission_id' in meta_df.columns:
                matched_meta_rows = meta_df.to_dict('records')
            else:
                diag['no_meta_rows_for_folder'] += 1
                continue

        mdf = pd.DataFrame(matched_meta_rows)
        if 'user_id' not in mdf.columns:
            diag['missing_user_id_column'] += 1
            continue

        # group by user_id
        for user_id, user_df in mdf.groupby('user_id'):
            user_df = user_df.copy()
            user_df['_time'] = user_df.apply(get_time_or_id, axis=1)
            user_df = user_df.sort_values(by='_time', na_position='last')
            accepted = user_df[user_df['status'] == 'Accepted']
            if accepted.empty:
                diag['no_accepted_for_user'] += 1
                continue
            fixed_row = accepted.iloc[0]
            fixed_id = str(fixed_row.get('submission_id', '')).strip()
            fixed_ext = str(fixed_row.get('filename_ext', '')).strip()
            fixed_code = read_submission_file(lang_dir, fixed_id, fixed_ext)
            if not fixed_code or len(fixed_code) < MIN_CODE_CHARS:
                diag['fixed_file_missing_or_short'] += 1
                continue

            # collect earlier non-accepted submissions
            seen = set()
            added = 0
            for _, r in user_df.iterrows():
                sid = str(r.get('submission_id', '')).strip()
                if sid == fixed_id:
                    break
                if r.get('status') == 'Accepted':
                    continue
                buggy_code = read_submission_file(lang_dir, sid, str(r.get('filename_ext', '')).strip())
                if not buggy_code or len(buggy_code) < MIN_CODE_CHARS:
                    diag['buggy_file_missing_or_short'] += 1
                    continue
                normb = normalize_code(buggy_code)
                if normb in seen:
                    diag['buggy_duplicate_skipped'] += 1
                    continue
                seen.add(normb)
                pairs.append({
                    'problem_id': pid,
                    'user_id': str(user_id),
                    'language': folder_lang,
                    'buggy_submission_id': sid,
                    'fixed_submission_id': fixed_id,
                    'status_buggy': str(r.get('status', '')),
                    'status_fixed': 'Accepted',
                    'buggy_code': buggy_code,
                    'fixed_code': fixed_code,
                    'test_cases': test_cases_for_problem
                })
                added += 1
                if MAX_PAIRS_PER_USER_PER_PROBLEM and added >= MAX_PAIRS_PER_USER_PER_PROBLEM:
                    break
            if added == 0:
                diag['no_earlier_nonaccepted'] += 1
            else:
                diag['pairs_added_for_user'] += added

    return pairs


# simple ASCII table printer (stdlib only)
def format_table(rows, headers):
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt.format(*headers), sep]
    for r in rows:
        lines.append(fmt.format(*[str(x) for x in r]))
    return "\n".join(lines)


def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    problem_ids = sorted(os.listdir(DATA_DIR))
    print(f"Found {len(problem_ids)} problem folders in {DATA_DIR}")

    start = time.time()
    total_pairs = 0
    problems_with_pairs = 0
    processed_problems = 0
    lang_counter = Counter()
    user_counter = Counter()
    diag = defaultdict(int)

    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        for idx, pid in enumerate(problem_ids, 1):
            if DRY_RUN and idx > DRY_RUN_LIMIT:
                break

            p_dir = os.path.join(DATA_DIR, pid)
            if not os.path.isdir(p_dir):
                diag['not_a_dir'] += 1
                continue
            meta_path = os.path.join(METADATA_DIR, f"{pid}.csv")
            if not os.path.exists(meta_path):
                diag['missing_metadata'] += 1
                continue
            meta_df = parse_metadata(meta_path)
            if meta_df is None:
                diag['bad_metadata_parse'] += 1
                continue

            pairs = extract_pairs_for_problem(pid, p_dir, meta_df, HTML_DIR, diag)
            if pairs:
                problems_with_pairs += 1
            for p in pairs:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                total_pairs += 1
                lang_counter[p['language']] += 1
                user_counter[p['user_id']] += 1
                if total_pairs % FLUSH_EVERY == 0:
                    fout.flush()
            processed_problems += 1
            if VERBOSE and idx % 200 == 0:
                elapsed = int(time.time() - start)
                print(f"→ Scanned {idx}/{len(problem_ids)} problems, written pairs so far: {total_pairs}, elapsed {elapsed}s")

    elapsed = int(time.time() - start)
    print("\n✅ Extraction complete.")
    print(f"Total problems scanned: {len(problem_ids)}")
    print(f"Problems processed (had metadata parsed): {processed_problems}")
    print(f"Problems that produced >=1 pair: {problems_with_pairs}")
    print(f"Total bug→fix pairs written: {total_pairs}")
    print(f"Elapsed time: {elapsed}s\n")

    # Language distribution
    lang_table = sorted(lang_counter.items(), key=lambda x: x[1], reverse=True)
    print("📊 Language-wise Distribution:")
    if lang_table:
        print(format_table(lang_table, ("Language (folder)", "Pair Count")))
        # save CSV
        df_lang = pd.DataFrame(lang_table, columns=["language", "pair_count"])
        df_lang.to_csv(LANGUAGE_DIST_CSV, index=False)
        print(f"Saved language distribution CSV to: {LANGUAGE_DIST_CSV}")
    else:
        print("(no pairs produced)")

    # Top users
    top_users = sorted(user_counter.items(), key=lambda x: x[1], reverse=True)[:100]
    print("\n👤 Top users (by pair count):")
    if top_users:
        print(format_table(top_users, ("User ID", "Pair Count")))
        df_users = pd.DataFrame(top_users, columns=["user_id", "pair_count"])
        df_users.to_csv(TOP_USERS_CSV, index=False)
        print(f"Saved top users CSV to: {TOP_USERS_CSV}")
    else:
        print("(no users)")

    # Diagnostics
    diag_rows = sorted(diag.items(), key=lambda x: x[1], reverse=True)
    print("\n🔍 Diagnostics summary:")
    if diag_rows:
        print(format_table(diag_rows, ("Reason", "Count")))
    else:
        print("(no diagnostics)")

    print(f"\nJSONL output written to: {OUT_FILE}")


if __name__ == "__main__":
    main()