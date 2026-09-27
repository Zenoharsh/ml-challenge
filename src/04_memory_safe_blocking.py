import os
import re
import sqlite3
import unicodedata
import pyarrow.parquet as pq
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DB_PATH = os.path.join(OUTPUT_DIR, "blocking_index.db")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

BATCH_SIZE = 25_000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKC", value)
    value = value.lower()

    # Replace punctuation with spaces
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value).strip()

    return value


def name_tokens(name):
    return [
        x for x in name.split()
        if len(x) >= 4
    ]


# ============================================================
# BLOCKING KEYS
# ============================================================

def generate_keys(name, address, country):

    name = normalize_text(name)
    address = normalize_text(address)
    country = normalize_text(country)

    keys = set()

    # --------------------------------------------------------
    # Exact normalized business name
    # --------------------------------------------------------

    if name:
        keys.add(
            ("NAME_EXACT", country, name)
        )

    # --------------------------------------------------------
    # First 6 characters of compact name
    # Helps with small spelling differences
    # --------------------------------------------------------

    compact = name.replace(" ", "")

    if len(compact) >= 6:
        keys.add(
            ("NAME_PREFIX6", country, compact[:6])
        )

    # --------------------------------------------------------
    # Individual strong name tokens
    # --------------------------------------------------------

    for token in name_tokens(name):

        # Avoid generic legal/business words
        if token in {
            "private",
            "limited",
            "company",
            "corporation",
            "services",
            "business",
            "group",
            "international",
            "holdings",
            "enterprise",
            "enterprises",
            "consulting"
        }:
            continue

        keys.add(
            ("NAME_TOKEN", country, token)
        )

    # --------------------------------------------------------
    # Address numbers
    # --------------------------------------------------------

    numbers = re.findall(
        r"\b\d{3,}\b",
        address
    )

    for number in numbers:

        keys.add(
            ("ADDRESS_NUMBER", country, number)
        )

    return keys


# ============================================================
# DATABASE SETUP
# ============================================================

def create_database():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(DB_PATH):
        print("Removing previous database...")
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-200000")

    conn.execute("""
        CREATE TABLE blocking_keys (
            key_type TEXT NOT NULL,
            country TEXT NOT NULL,
            key_value TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source TEXT NOT NULL
        )
    """)

    conn.commit()

    return conn


# ============================================================
# STREAM PARQUET INTO SQLITE
# ============================================================

def index_source(conn, parquet_path, source):

    print()
    print("=" * 60)
    print(f"Indexing {source}")
    print(parquet_path)
    print("=" * 60)

    parquet = pq.ParquetFile(parquet_path)

    total_rows = parquet.metadata.num_rows

    processed = 0

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ):

        df = batch.to_pandas()

        rows = []

        for row in df.itertuples(index=False):

            entity_id = str(row.entity_id)

            keys = generate_keys(
                row.business_name,
                row.business_address,
                row.country
            )

            for key_type, country, key_value in keys:

                rows.append(
                    (
                        key_type,
                        country,
                        key_value,
                        entity_id,
                        source
                    )
                )

        if rows:

            conn.executemany(
                """
                INSERT INTO blocking_keys
                (
                    key_type,
                    country,
                    key_value,
                    entity_id,
                    source
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                rows
            )

        conn.commit()

        processed += len(df)

        print(
            f"{source}: "
            f"{processed:,}/{total_rows:,} "
            f"({processed / total_rows * 100:.1f}%)"
        )

        del df
        del rows

    print(f"Finished {source}")


# ============================================================
# SQLITE INDEX
# ============================================================

def create_indexes(conn):

    print()
    print("Creating SQLite index...")

    conn.execute("""
        CREATE INDEX idx_block_key
        ON blocking_keys(
            key_type,
            country,
            key_value
        )
    """)

    conn.execute("""
        CREATE INDEX idx_entity
        ON blocking_keys(entity_id)
    """)

    conn.commit()

    print("SQLite indexes created.")


# ============================================================
# BUILD
# ============================================================

def build_index():

    conn = create_database()

    try:

        index_source(
            conn,
            os.path.join(
                TRAIN_DIR,
                "train_source2.parquet"
            ),
            "S2"
        )

        index_source(
            conn,
            os.path.join(
                TRAIN_DIR,
                "train_source3.parquet"
            ),
            "S3"
        )

        create_indexes(conn)

    finally:

        conn.close()

    print()
    print("=" * 60)
    print("BLOCKING INDEX COMPLETE")
    print("=" * 60)
    print(DB_PATH)


if __name__ == "__main__":
    build_index()