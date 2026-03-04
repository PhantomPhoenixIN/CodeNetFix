# On the Role of Temporal Structure in Historic Repair Trajectories for Automatic Program Repair

This repository accompanies the paper:

**On the Role of Temporal Structure in Historic Repair Trajectories for Automatic Program Repair**

This work studies whether **preserving the chronological order of human debugging attempts improves neural Automatic Program Repair (APR).**

Most APR datasets treat bugs as **single buggy–fix pairs**, ignoring the iterative nature of human debugging. However, programmers usually refine programs through **multiple submissions before reaching a correct solution**.

This project leverages **historic repair trajectories** mined from **Project CodeNet** to investigate whether **temporal ordering provides additional learning signal for neural repair models.**

---

# Key Idea

Human debugging is inherently **iterative**.

Example repair trajectory:

Submission 1 → Wrong Answer
Submission 2 → Time Limit Exceeded
Submission 3 → Accepted


Each step represents a **progressively refined program**.

We hypothesize that **preserving this temporal structure improves neural repair models beyond simply increasing dataset size.**

---

# Repair Trajectory Illustration

Below is an example illustrating how programmers progressively refine solutions through multiple submissions.

<p align="center">
  <img src="Program Submission.png" width="900">
</p>

<p align="center">
  <em>
    Execution-grounded quality-stratified code translation framework.
    Multilingual source programs from Project CodeNet are translated using
    multiple teacher models and stratified based on AST parsability,
    compilability, and functional correctness.
  </em>
</p>

Each submission receives execution feedback (Wrong Answer, Time Limit Exceeded, etc.), guiding iterative refinement until an **Accepted solution** is obtained.

---

# Dataset: CodeNetFix

We construct **CodeNetFix**, a large-scale dataset of repair trajectories mined from **Project CodeNet**.

Submissions are grouped by:

(problem_id, user_id, language)


and ordered chronologically.

Each trajectory consists of all buggy submissions before the **first Accepted submission**.

---

# Dataset Variants

To isolate the effect of temporal ordering we construct three dataset variants.

| Dataset | Description | Temporal Order |
|-------|-------------|---------------|
| CodeNetFix-RF | One randomly sampled bug–fix pair per trajectory | ❌ |
| CodeNetFix-HTS | All bug–fix pairs but shuffled | ❌ |
| CodeNetFix-HT | All bug–fix pairs preserving chronological order | ✅ |

This allows controlled comparison between:

- single-step supervision
- multi-step supervision without order
- temporally ordered repair trajectories

---

# Dataset Statistics

| Dataset | Instances | Trajectories | Avg Pairs / Trajectory | Temporal Order |
|-------|-----------|-------------|-----------------------|---------------|
| CodeNetFix-RF | 1,151,142 | 1,151,142 | 1.00 | No |
| CodeNetFix-HTS | 2,137,707 | 1,151,142 | 1.86 | Shuffled |
| CodeNetFix-HT | 2,137,707 | 1,151,142 | 1.86 | Preserved |

Additional statistics:

- **2,721 programming problems**
- **102,008 users**
- **55 programming languages**

---

# Programming Language Distribution

| Language | Instances | Percentage |
|--------|----------|-----------|
| C++ | 1,234,311 | 57.74% |
| Python | 504,461 | 23.60% |
| C | 120,930 | 5.66% |
| Java | 116,201 | 5.44% |
| C# | 36,878 | 1.73% |
| Others | 124,927 | 5.83% |

---

# Model

All experiments use **CodeT5-base (220M parameters)**.

Program repair is formulated as **code-to-code translation**.

Input prompt format:

Task:

Given a buggy program and its problem description, generate a corrected version.

Problem Description:

{description}

Buggy Code:

{buggy_code}

Fixed Code:


Training setup:

- Full fine-tuning
- identical hyperparameters across experiments
- greedy decoding during inference

---

# Evaluation

Performance is measured using **execution-based correctness**.

### Pass@1

A repair is considered correct only if:

- it compiles successfully
- it passes all test cases

Pass@1 = (# successful repairs) / (total test programs)


Test set size:

109,579 programs


---

# Overall Results

| Model | Pass@1 | Additional Repairs |
|------|-------|-------------------|
| CodeNetFix-RF | 36.19% | — |
| CodeNetFix-HTS | 38.15% | +2,158 |
| CodeNetFix-HT | **39.44%** | **+3,563** |

Key finding:

**Preserving temporal order improves Pass@1 by 1.29 percentage points compared to shuffled trajectories.**

This corresponds to **1,405 additional correct repairs.**

---

# Key Results

## Overall Repair Performance

![Overall Performance](figures/overall_results.png)

Temporally ordered trajectories consistently outperform both random and shuffled variants.

---

## Stage-wise Repair Analysis

![Stage-wise Results](figures/stage_results.png)

Temporal ordering improves repair performance across:

- early-stage debugging
- intermediate refinement
- late-stage corrections

---

## Learning Curve Analysis

![Learning Curve](figures/learning_curve.png)

The advantage of temporal ordering **increases as training data grows**.

---

# Language-wise Results

| Language | RF | HTS | HT |
|---------|----|----|----|
| C++ | 33.68% | 35.58% | **37.29%** |
| Java | 40.26% | 41.85% | **44.12%** |
| Python | 41.00% | 43.15% | **43.27%** |

Temporal ordering improves repair performance across all major languages.

---

# Error-Type Analysis

| Error Type | RF | HTS | HT |
|-----------|----|----|----|
| Wrong Answer | 37.28% | 40.56% | **41.88%** |
| Compile Error | 33.11% | 34.41% | **36.09%** |
| Runtime Error | 28.13% | 29.49% | 29.37% |
| Time Limit Exceeded | 41.70% | 45.12% | **47.76%** |

The largest gains occur for:

- **semantic errors (Wrong Answer)**
- **algorithmic errors (Time Limit Exceeded)**

---

# Key Takeaways

1. Human debugging is **iterative rather than single-step**.

2. Historic repair trajectories encode **structured correction patterns**.

3. Preserving chronological order improves neural program repair **without modifying model architecture**.

4. Data organization itself can act as a **supervision signal**.

---

# Repository Structure

