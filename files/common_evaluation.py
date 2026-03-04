#!/usr/bin/env python3

"""
PASS@1 EVALUATION WITH STAGE ANALYSIS

Includes:
- Per-language performance
- Stage-wise performance
- Running stats every 50 instances (HPC-safe flush)
- Final structured stats dictionary for paper reporting
"""

# =====================================================
# IMPORTS
# =====================================================

import json
import subprocess
import tempfile
import os
import shutil
import ast
import javalang
from collections import defaultdict

# =====================================================
# CONFIG
# =====================================================

# Change this path to evaluate any model prediction file
INPUT_JSONL = "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets/rectifier_predictions/TLSA/CodeNetFix_RF_test_HTS_len1_greedy.jsonl"
PROBLEM_TESTS_FILE = "/home/swaminathanj/TransRectify_New/NMT_Rectifier/content/datasets/problem_tests.json"

EXEC_TIMEOUT = 5
PROGRESS_INTERVAL = 10

# =====================================================
# LOADERS
# =====================================================

def load_tests(path):
    tests = {}
    with open(path, "r", encoding="utf-8") as f:
        for item in json.load(f):
            tests[item["problem_id"]] = {
                "input": item["input"],
                "output": item["output"].strip()
            }
    return tests

# =====================================================
# PYTHON
# =====================================================

def python_parsable(code):
    try:
        ast.parse(code)
        return True
    except:
        return False

def run_python(code, test_input):
    tmpdir = tempfile.mkdtemp()
    py_file = os.path.join(tmpdir, "main.py")

    with open(py_file, "w") as f:
        f.write(code)

    try:
        proc = subprocess.run(
            ["python3", py_file],
            input=test_input,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=EXEC_TIMEOUT
        )
        return proc.stdout.strip()
    except:
        return None
    finally:
        shutil.rmtree(tmpdir)

# =====================================================
# C++
# =====================================================

def normalize_cpp(code):
    if "#include" not in code:
        return "#include <bits/stdc++.h>\nusing namespace std;\n\n" + code
    return code

def cpp_compile(code):
    code = normalize_cpp(code)
    tmpdir = tempfile.mkdtemp()
    cpp_file = os.path.join(tmpdir, "main.cpp")
    exe_file = os.path.join(tmpdir, "a.out")

    with open(cpp_file, "w") as f:
        f.write(code)

    proc = subprocess.run(
        ["g++", cpp_file, "-O2", "-std=c++17", "-o", exe_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if proc.returncode != 0:
        shutil.rmtree(tmpdir)
        return False, None

    return True, tmpdir

def run_cpp(tmpdir, test_input):
    try:
        proc = subprocess.run(
            [os.path.join(tmpdir, "a.out")],
            input=test_input,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=EXEC_TIMEOUT
        )
        return proc.stdout.strip()
    except:
        return None

# =====================================================
# JAVA
# =====================================================

def java_parsable(code):
    try:
        javalang.parse.parse(code)
        return True
    except:
        return False

def java_compile(code):
    tmpdir = tempfile.mkdtemp()
    java_file = os.path.join(tmpdir, "Main.java")

    with open(java_file, "w") as f:
        f.write(code)

    proc = subprocess.run(
        ["javac", java_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if proc.returncode != 0:
        shutil.rmtree(tmpdir)
        return False, None

    return True, tmpdir

def run_java(tmpdir, test_input):
    try:
        proc = subprocess.run(
            ["java", "-cp", tmpdir, "Main"],
            input=test_input,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=EXEC_TIMEOUT
        )
        return proc.stdout.strip()
    except:
        return None

# =====================================================
# EVALUATION
# =====================================================

def evaluate_instance(code, lang, test):

    if not code or not code.strip():
        return False, False, False

    lang = lang.lower()

    if lang == "python":
        if not python_parsable(code):
            return False, False, False
        output = run_python(code, test["input"])
        return True, True, output == test["output"]

    if lang in {"c++", "cpp"}:
        compilable, tmpdir = cpp_compile(code)
        if not compilable:
            return False, False, False
        output = run_cpp(tmpdir, test["input"])
        shutil.rmtree(tmpdir)
        return True, True, output == test["output"]

    if lang == "java":
        if not java_parsable(code):
            return False, False, False
        compilable, tmpdir = java_compile(code)
        if not compilable:
            return True, False, False
        output = run_java(tmpdir, test["input"])
        shutil.rmtree(tmpdir)
        return True, True, output == test["output"]

    return False, False, False

# =====================================================
# MAIN
# =====================================================

def evaluate():

    TESTS = load_tests(PROBLEM_TESTS_FILE)

    global_stats = defaultdict(int)
    lang_stats = defaultdict(lambda: defaultdict(int))
    stage_stats = defaultdict(lambda: defaultdict(int))

    with open(INPUT_JSONL, "r") as fin:

        for idx, line in enumerate(fin, 1):

            item = json.loads(line)

            pid = item["problem_id"]
            lang = item.get("language", "UNKNOWN")
            code = item.get("prediction", "")
            stage = item.get("repair_stage", "unknown")

            test = TESTS.get(pid)
            if test is None:
                continue

            global_stats["total"] += 1
            lang_stats[lang]["total"] += 1
            stage_stats[stage]["total"] += 1

            parsable, compilable, functional = evaluate_instance(code, lang, test)

            if parsable:
                global_stats["parsable"] += 1
                lang_stats[lang]["parsable"] += 1
                stage_stats[stage]["parsable"] += 1

            if compilable:
                global_stats["compilable"] += 1
                lang_stats[lang]["compilable"] += 1
                stage_stats[stage]["compilable"] += 1

            if functional:
                global_stats["pass@1"] += 1
                lang_stats[lang]["pass@1"] += 1
                stage_stats[stage]["pass@1"] += 1

            # ===============================
            # RUNNING STATS EVERY 50
            # ===============================
            if idx % PROGRESS_INTERVAL == 0:

                total = global_stats["total"]
                parsable = global_stats["parsable"]
                compilable = global_stats["compilable"]
                pass1 = global_stats["pass@1"]

                print("\nProcessed:", total)
                print(f"Parsable: {parsable} ({parsable/total*100:.2f}%)")
                print(f"Compilable: {compilable} ({compilable/total*100:.2f}%)")
                print(f"Pass@1: {pass1} ({pass1/total*100:.2f}%)\n", flush=True)

    # FINAL SUMMARY
    print("\n" + "="*80)
    print("GLOBAL PERFORMANCE")
    print("="*80)

    total = global_stats["total"]
    print(f"Total: {total}")
    print(f"Parsable: {global_stats['parsable']} ({global_stats['parsable']/total*100:.2f}%)")
    print(f"Compilable: {global_stats['compilable']} ({global_stats['compilable']/total*100:.2f}%)")
    print(f"Pass@1: {global_stats['pass@1']} ({global_stats['pass@1']/total*100:.2f}%)")

    return {
        "global": dict(global_stats),
        "per_language": {k: dict(v) for k, v in lang_stats.items()},
        "per_stage": {k: dict(v) for k, v in stage_stats.items()},
    }


if __name__ == "__main__":
    results = evaluate()