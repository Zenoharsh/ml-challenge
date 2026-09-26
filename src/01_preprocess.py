import os
import shutil
import glob
import re
import pandas as pd
import string

# Define paths relative to the script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDENT_RESOURCE_TRAIN = os.path.join(BASE_DIR, "student_resource", "dataset", "train")
STUDENT_RESOURCE_TEST = os.path.join(BASE_DIR, "student_resource", "dataset", "test")
DATA_TRAIN = os.path.join(BASE_DIR, "data", "train")
DATA_TEST = os.path.join(BASE_DIR, "data", "test")

def move_files():
    print("Moving files from student_resource to data folder...")
    
    # Move train files
    if os.path.exists(STUDENT_RESOURCE_TRAIN):
        for file_path in glob.glob(os.path.join(STUDENT_RESOURCE_TRAIN, "*.tsv")):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(DATA_TRAIN, filename)
            if not os.path.exists(dest_path):
                shutil.move(file_path, dest_path)
                print(f"Moved {filename} to {DATA_TRAIN}")
            else:
                print(f"{filename} already exists in {DATA_TRAIN}")
    else:
        print(f"Warning: {STUDENT_RESOURCE_TRAIN} does not exist.")
    
    # Move test files
    if os.path.exists(STUDENT_RESOURCE_TEST):
        for file_path in glob.glob(os.path.join(STUDENT_RESOURCE_TEST, "*.tsv")):
            filename = os.path.basename(file_path)
            dest_path = os.path.join(DATA_TEST, filename)
            if not os.path.exists(dest_path):
                shutil.move(file_path, dest_path)
                print(f"Moved {filename} to {DATA_TEST}")
            else:
                print(f"{filename} already exists in {DATA_TEST}")
    else:
        print(f"Warning: {STUDENT_RESOURCE_TEST} does not exist.")

def normalize_text(text):
    if pd.isna(text):
        return text
    
    # Lowercase
    text = str(text).lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Standardize common business/address abbreviations
    abbreviations = {
        r'\bcorporation\b': 'corp',
        r'\bincorporated\b': 'inc',
        r'\bcompany\b': 'co',
        r'\blimited\b': 'ltd',
        r'\bstreet\b': 'st',
        r'\bavenue\b': 'ave',
        r'\bboulevard\b': 'blvd',
        r'\broad\b': 'rd',
        r'\bdrive\b': 'dr',
        r'\blane\b': 'ln',
        r'\bplace\b': 'pl',
        r'\bsuite\b': 'ste',
        r'\bapartment\b': 'apt',
        r'\bdepartment\b': 'dept'
    }
    
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)
        
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_data():
    print("Processing data...")
    # Define files to process
    train_files = ['train_source1.tsv', 'train_source2.tsv', 'train_source3.tsv', 'train_ground_truth.tsv']
    test_files = ['test_source1.tsv', 'test_source2.tsv', 'test_source3.tsv']
    
    all_files = [(f, DATA_TRAIN) for f in train_files] + [(f, DATA_TEST) for f in test_files]
    
    for filename, folder in all_files:
        file_path = os.path.join(folder, filename)
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping...")
            continue
            
        print(f"\nLoading {filename}...")
        df = pd.read_csv(file_path, sep='\t', dtype=str, on_bad_lines='skip') 
        
        columns_to_normalize = ['business_name', 'business_address']
        normalized_any = False
        for col in columns_to_normalize:
            if col in df.columns:
                print(f"Normalizing column '{col}' in {filename}...")
                df[col] = df[col].apply(normalize_text)
                normalized_any = True
                
        # Save as parquet regardless of whether it had the specific columns, 
        # as parquet is better for subsequent steps
        parquet_filename = filename.replace('.tsv', '.parquet')
        parquet_path = os.path.join(folder, parquet_filename)
        print(f"Saving to {parquet_path}...")
        df.to_parquet(parquet_path, index=False)

if __name__ == "__main__":
    move_files()
    process_data()
    print("\nPreprocessing completed successfully.")
