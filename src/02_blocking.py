import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_TRAIN = os.path.join(BASE_DIR, "data", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def create_combined_text(df):
    """Concatenate business_name and business_address to form a single text string per record."""
    name = df['business_name'].fillna('')
    address = df['business_address'].fillna('')
    return name + " " + address

def generate_candidate_pairs(k=10):
    print("Loading data...")
    # Load source files
    df1_path = os.path.join(DATA_TRAIN, "train_source1.parquet")
    df2_path = os.path.join(DATA_TRAIN, "train_source2.parquet")
    df3_path = os.path.join(DATA_TRAIN, "train_source3.parquet")
    
    if not all(os.path.exists(p) for p in [df1_path, df2_path, df3_path]):
        raise FileNotFoundError("One or more train_source parquet files are missing. Run 01_preprocess.py first.")

    df1 = pd.read_parquet(df1_path)
    df2 = pd.read_parquet(df2_path)
    df3 = pd.read_parquet(df3_path)

    print("Concatenating business_name and business_address columns...")
    df1['combined'] = create_combined_text(df1)
    df2['combined'] = create_combined_text(df2)
    df3['combined'] = create_combined_text(df3)

    # Combine df2 and df3 to form the candidate pool
    print("Combining Source 2 and Source 3 into a single candidate pool...")
    df_pool = pd.concat([df2, df3], ignore_index=True)
    
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    
    print("Vectorizing text using HashingVectorizer + TF-IDF (to prevent MemoryError)...")
    # HashingVectorizer avoids building a dictionary in memory, saving significant RAM
    # We use word-level tokens (instead of characters) to massively reduce the number of tokens 
    # per document, which avoids hitting NumPy ArrayMemoryError on the sparse indices.
    hasher = HashingVectorizer(analyzer='word', ngram_range=(1, 2), n_features=100000, alternate_sign=False, dtype=np.float32)
    tfidf = TfidfTransformer()
    
    print("Hashing pool data...")
    X_pool_counts = hasher.transform(df_pool['combined'])
    print("Applying TF-IDF to pool data...")
    X_pool = tfidf.fit_transform(X_pool_counts)
    
    print("Hashing Source 1 data...")
    X_source1_counts = hasher.transform(df1['combined'])
    print("Applying TF-IDF to Source 1 data...")
    X_source1 = tfidf.transform(X_source1_counts)

    print(f"Building NearestNeighbors index for {X_pool.shape[0]} pool entities...")
    # Brute force with cosine metric on sparse matrices is generally the fastest exact method in sklearn
    nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1)
    nn.fit(X_pool)

    print(f"Searching for top {k} neighbors for {X_source1.shape[0]} Source 1 entities...")
    distances, indices = nn.kneighbors(X_source1)

    print("Formatting output...")
    candidate_records = []
    
    pool_entity_ids = df_pool['entity_id'].values
    source1_entity_ids = df1['entity_id'].values

    for i in range(len(source1_entity_ids)):
        s1_id = source1_entity_ids[i]
        c_ids = [pool_entity_ids[idx] for idx in indices[i]]
        # Empty candidate lists are left blank (though kneighbors usually returns exactly k unless pool < k)
        c_ids_str = ",".join(c_ids) if c_ids else ""
        candidate_records.append({
            'source1_entity_id': s1_id,
            'candidate_entity_ids': c_ids_str
        })

    out_df = pd.DataFrame(candidate_records)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    print(f"Saving to {out_path}...")
    out_df.to_csv(out_path, sep='\t', index=False)
    print("Candidate generation complete.")

if __name__ == "__main__":
    generate_candidate_pairs(k=10)
