## IndicParam: Benchmark to Evaluate LLMs on Low-Resource Indic Languages

IndicParam is a human-curated benchmark of graduate-level multiple-choice questions (MCQs) for evaluating Large Language Models (LLMs) on **low- and extremely low-resource Indic languages**, following the design of ParamBench but extending it beyond Hindi.

### Overview

- **Goal**: Assess LLMs on language understanding and general knowledge in under-represented Indic languages under a unified, exam-style MCQ setting.
- **Source**: Official UGC-NET language question papers and answer keys, collected across multiple years and sessions.
- **Scale**: The benchmark described in the paper contains **13,207 questions** across **11 languages**, plus a separate Sanskrit–English code-mixed set; this repository follows the same schema and may include updated or extended releases.

### Languages and Scripts

IndicParam covers Indic languages that are low-resource or extremely low-resource in web-scale pretraining corpora:

- **Low-resource (4)**: Nepali, Gujarati, Marathi, Odia  
- **Extremely low-resource (7)**: Dogri, Maithili, Rajasthani, Sanskrit, Bodo, Santali, Konkani  
- **Code-mixed set**: Sanskrit–English (Sans-Eng)

Scripts:

- **Devanagari**: Nepali, Marathi, Maithili, Konkani, Bodo, Dogri, Rajasthani, Sanskrit  
- **Gujarati**: Gujarati  
- **Odia (Orya)**: Odia  
- **Ol Chiki (Olck)**: Santali  

The benchmark explicitly targets languages with **very limited web presence** despite large speaker populations, making them challenging for current LLMs.

### Dataset and Annotations

The main release in this repository is provided as `data.csv`, with one row per question. Core fields (column names may differ slightly from the paper notation) include:

- **subject**: Language / subject (e.g., `Nepali`, `Maithili`, `Sanskrit`).
- **exam_name**, **paper_number**, **question_number**: UGC-NET metadata.
- **question_text**: Full question text in the target language (or code-mixed variant).
- **option_a**, **option_b**, **option_c**, **option_d**: Answer options, in the same language.
- **correct_answer**: Gold label (`a`/`b`/`c`/`d`).
- **unique_question_id**: Stable identifier for tracking and de-duplication.
- **question_type**: Encodes the question format (see below).
- **difficulty_level**: Difficulty category (e.g., `Easy`, `Medium`, `Hard`), as used in the paper.
- **classification_response / related fields**: JSON-style or derived annotations for question class (language understanding vs. general knowledge) and other metadata.

Each instance is a **graduate-level, human-authored MCQ** drawn from official exams. Content spans:

- **Language understanding (LU)**: Linguistics and grammar (morphology, syntax, semantics, discourse).
- **General knowledge (GK)**: Facts, world knowledge, literature, history, and culture.

Questions are also classified into the six formats used in the paper:

- **Multiple-choice**  
- **Assertion & Reason**  
- **List Matching**  
- **Fill in the Blanks**  
- **Identify Incorrect Statement**  
- **Ordering**

All questions are intended **only for evaluation**, not for model training.

### Task and Evaluation Protocol

The benchmark is framed as a **multiple-choice question answering** task:

- **Input**: Question text and four options (A–D), in the original language/script.
- **Output**: A single option label (`A`, `B`, `C`, or `D`) with no explanation.
- **Primary metric**: **Accuracy** (percentage of correctly answered questions), reported:
  - Per language
  - Separately for **LU** vs. **GK**
  - By **question type** (MCQ, A\&R, list matching, etc.), as in the paper

For comparability with the paper:

- Use **zero-shot** prompting with a consistent instruction template across languages.
- Use **deterministic decoding** (e.g., temperature 0, `do_sample=False`) and a short max token budget suitable for single-letter responses.

### Repository Contents and Usage

- **`data.csv`**: Main benchmark file in CSV format (UTF-8).  
- **`evaluate_open_models.py`**: Example script to evaluate open-weight Hugging Face models on IndicParam.  
- **`evaluate_gpt_oss.py`**: Example script to run the GPT-OSS-120B model on the same data.  
- **`evaluate_openrouter.py`**: Example script to benchmark closed models via the OpenRouter API.

Typical usage pattern:

- **Prepare environment**: Install Python dependencies (see `requirements.txt` if present) and configure any required API keys or model caches.
- **Run evaluation**: Invoke one of the scripts with your chosen model configuration and an output directory; the scripts will:
  - Load `data.csv`
  - Construct language-aware MCQ prompts
  - Record model predictions and compute accuracy

Script-level arguments and options are documented via the `-h`/`--help` flags within each script.

### Citation

If you use IndicParam in your research or system evaluations, please cite the accompanying paper:

```bibtex
@article{maheshwari2025indicparam,
  title   = {IndicParam: Benchmark to Evaluate LLMs on Low-Resource Indic Languages},
  author  = {Maheshwari, Ayush and Sharma, Kaushal and Patel, Vivek and Maheshwari, Aditya},
  journal = {arXiv preprint arXiv:2508.16185},
  year    = {2025}
}
```

Please also cite ParamBench where relevant for Hindi and for the question-type taxonomy.

### License and Ethics

IndicParam is released for **non-commercial research and evaluation**. The questions are sourced from publicly available UGC-NET language papers; annotation was conducted by native speakers as described in the paper.  
For any commercial or large-scale downstream use, please consult the UGC-NET/UGC–NTA terms and seek appropriate permissions.
