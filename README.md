<div align="center">
  
[![arXiv](https://img.shields.io/badge/arXiv-2512.00333-f9f107.svg)](https://arxiv.org/pdf/2512.00333)
[![huggingface](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/bharatgenai/IndicParam)
</div>

## IndicParam: Benchmark to evaluate LLMs on low-resource Indic Languages
IndicParam is a human-curated benchmark of >13,000 questions for evaluating LLMs on **low- and extremely low-resource Indic languages**. We evaluated 19 LLMs, both proprietary and open-weights, which reveals that even the top-performing GPT-5 reaches only 45% average accuracy, followed by DeepSeek-3.2 (43.1) and Claude-4.5 (42.7).

### Overview

- The benchmark contains **13,207 questions** across **11 languages**, plus a separate Sanskrit–English code-mixed set; this repository.
- **Source**: Official UGC-NET language question papers and answer keys, collected across multiple years and sessions.


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

### Dataset and Annotations

The main release in this repository is provided as `data.csv`, with one row per question. Core fields (column names may differ slightly from the paper notation) include:

- **subject**: Language / subject (e.g., `Nepali`, `Maithili`, `Sanskrit`).
- **exam_name**, **paper_number**, **question_number**: UGC-NET metadata.
- **question_text**: Full question text in the target language (or code-mixed variant).
- **option_a**, **option_b**, **option_c**, **option_d**: Answer options, in the same language.
- **correct_answer**: Gold label (`a`/`b`/`c`/`d`).
- **unique_question_id**: Stable identifier for tracking and de-duplication.
- **question_type**: Encodes the question format (see below).

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
@misc{maheshwari2025indicparambenchmarkevaluatellms,
      title={IndicParam: Benchmark to evaluate LLMs on low-resource Indic Languages}, 
      author={Ayush Maheshwari and Kaushal Sharma and Vivek Patel and Aditya Maheshwari},
      year={2025},
      eprint={2512.00333},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.00333}, 
}
```

