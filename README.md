# ML Challenge Pipeline

This repository contains the data preprocessing and entity blocking pipeline for the ML Challenge.

## Pipeline Steps

### 1. Data Preprocessing (`src/01_preprocess.py`)
- Moves `.tsv` dataset files into the standard `data/train/` and `data/test/` directory structure.
- Cleans and normalizes `business_name` and `business_address` columns:
  - Lowercases strings and removes punctuation.
  - Standardizes common abbreviations (e.g., `corporation` to `corp`, `street` to `st`).
  - Removes extra whitespace.
- Saves the cleaned datasets as `.parquet` files for faster loading in subsequent steps.

### 2. Candidate Blocking (`src/02_blocking.py`)
- Concatenates the normalized `business_name` and `business_address` for each record.
- Employs a highly memory-efficient TF-IDF vectorization strategy using `HashingVectorizer` with word n-grams (1-2) to avoid OOM memory errors on the large dataset.
- Uses exact Nearest Neighbors (via scikit-learn with cosine distance) to retrieve the top 10 most similar candidate entities from Source 2 and Source 3 for each entity in Source 1.
- Outputs the generated candidate pairs to `output/candidate_pairs.tsv`.

## Setup

1. Ensure the dataset is present in `student_resource/dataset/` or directly inside `data/train/` and `data/test/`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python src/01_preprocess.py
   python src/02_blocking.py
   ```
