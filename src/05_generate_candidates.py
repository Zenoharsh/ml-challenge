import os
import re
import sqlite3
import unicodedata
import pyarrow.parquet as pq


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "data",
    "train"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

DB_PATH = os.path.join(
    OUTPUT_DIR,
    "blocking_index.db"
)

SOURCE1_PATH = os.path.join(
    TRAIN_DIR,
    "train_source1.parquet"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)

# Safe settings for i3 laptop
BATCH_SIZE = 2_000
TOP_K = 50
MAX_RECORDS = 2_000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):

    if value is None:
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = value.lower()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    ).strip()

    return value


# ============================================================
# BLOCKING KEYS
# ============================================================

def generate_keys(name, address, country):

    name = normalize_text(name)
    address = normalize_text(address)
    country = normalize_text(country)

    keys = set()

    # Exact normalized business name
    if name:
        keys.add(
            (
                "NAME_EXACT",
                country,
                name
            )
        )

    # First 6 characters of compact name
    compact = name.replace(" ", "")

    if len(compact) >= 6:

        keys.add(
            (
                "NAME_PREFIX6",
                country,
                compact[:6]
            )
        )

    # Distinctive name tokens
    ignored = {
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
    }

    for token in name.split():

        if (
            len(token) >= 4
            and token not in ignored
        ):

            keys.add(
                (
                    "NAME_TOKEN",
                    country,
                    token
                )
            )

    # Address numbers
    numbers = re.findall(
        r"\b\d{3,}\b",
        address
    )

    for number in numbers:

        keys.add(
            (
                "ADDRESS_NUMBER",
                country,
                number
            )
        )

    return keys


# ============================================================
# MAIN
# ============================================================

def generate_candidates():

    if not os.path.exists(DB_PATH):

        raise FileNotFoundError(
            f"Blocking database not found:\n{DB_PATH}"
        )

    if not os.path.exists(SOURCE1_PATH):

        raise FileNotFoundError(
            f"Source 1 file not found:\n{SOURCE1_PATH}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 65)
    print("FAST CANDIDATE GENERATION")
    print("=" * 65)

    print(f"Database : {DB_PATH}")
    print(f"Source 1 : {SOURCE1_PATH}")
    print(f"Output   : {OUTPUT_PATH}")
    print(f"Batch    : {BATCH_SIZE:,}")
    print(f"Top-K    : {TOP_K}")
    print(f"Max test records : {MAX_RECORDS:,}")

    # --------------------------------------------------------
    # Open SQLite database
    # --------------------------------------------------------

    conn = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True
    )

    conn.execute(
        "PRAGMA cache_size = -500000"
    )

    conn.execute(
        "PRAGMA temp_store = MEMORY"
    )

    # --------------------------------------------------------
    # Check database
    # --------------------------------------------------------

    tables = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
        """
    ).fetchall()

    print("\nSQLite tables:")

    for table in tables:

        print(
            " ",
            table[0]
        )

    # --------------------------------------------------------
    # Temporary table for Source 1 blocking keys
    # --------------------------------------------------------

    conn.execute(
        """
        CREATE TEMP TABLE IF NOT EXISTS query_keys (
            row_no INTEGER,
            key_type TEXT,
            country TEXT,
            key_value TEXT,
            PRIMARY KEY (
                row_no,
                key_type,
                country,
                key_value
            )
        )
        """
    )

    # --------------------------------------------------------
    # Read Source 1 through PyArrow
    # --------------------------------------------------------

    parquet = pq.ParquetFile(
        SOURCE1_PATH
    )

    total = min(
        parquet.metadata.num_rows,
        MAX_RECORDS
    )

    print(
        f"\nSource 1 records: {total:,}"
    )

    processed = 0

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as output:

        output.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        # ----------------------------------------------------
        # Process batches
        # ----------------------------------------------------

        for batch in parquet.iter_batches(

            batch_size=BATCH_SIZE,

            columns=[
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]
        ):

            # Only process MAX_RECORDS
            remaining = (
                MAX_RECORDS
                - processed
            )

            if remaining <= 0:
                break

            rows = batch.to_pylist()[
                :remaining
            ]

            # ------------------------------------------------
            # Clear temporary keys
            # ------------------------------------------------

            conn.execute(
                "DELETE FROM query_keys"
            )

            key_rows = []

            # ------------------------------------------------
            # Generate blocking keys
            # ------------------------------------------------

            for row_no, row in enumerate(rows):

                keys = generate_keys(

                    row["business_name"],

                    row["business_address"],

                    row["country"]
                )

                for (
                    key_type,
                    country,
                    key_value
                ) in keys:

                    key_rows.append(
                        (
                            row_no,
                            key_type,
                            country,
                            key_value
                        )
                    )

            # ------------------------------------------------
            # Insert keys for this batch
            # ------------------------------------------------

            conn.executemany(

                """
                INSERT OR IGNORE INTO query_keys
                (
                    row_no,
                    key_type,
                    country,
                    key_value
                )
                VALUES (?, ?, ?, ?)
                """,

                key_rows
            )

            # ------------------------------------------------
            # Batch JOIN against blocking index
            # ------------------------------------------------

            query = """

            WITH ranked AS (

                SELECT

                    q.row_no,

                    b.entity_id,

                    COUNT(*) AS score,

                    ROW_NUMBER() OVER (

                        PARTITION BY q.row_no

                        ORDER BY
                            COUNT(*) DESC,
                            b.entity_id

                    ) AS rn

                FROM query_keys q

                JOIN blocking_keys b

                  ON b.key_type = q.key_type

                 AND b.country = q.country

                 AND b.key_value = q.key_value

                GROUP BY

                    q.row_no,

                    b.entity_id
            )

            SELECT

                row_no,

                entity_id

            FROM ranked

            WHERE rn <= ?

            ORDER BY

                row_no,

                rn

            """

            results = conn.execute(
                query,
                (TOP_K,)
            ).fetchall()

            # ------------------------------------------------
            # Convert results into candidate lists
            # ------------------------------------------------

            candidates = {}

            for (
                row_no,
                entity_id
            ) in results:

                candidates.setdefault(
                    row_no,
                    []
                ).append(
                    str(entity_id)
                )

            # ------------------------------------------------
            # Write every Source 1 record
            # ------------------------------------------------

            for row_no, row in enumerate(rows):

                source1_id = str(
                    row["entity_id"]
                )

                candidate_ids = candidates.get(
                    row_no,
                    []
                )

                output.write(

                    source1_id
                    + "\t"
                    + ",".join(candidate_ids)
                    + "\n"
                )

            processed += len(rows)

            percentage = (
                processed
                / total
                * 100
            )

            print(
                f"Processed "
                f"{processed:,}/"
                f"{total:,} "
                f"({percentage:.2f}%)"
            )

    conn.close()

    print()
    print("=" * 65)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 65)

    print(
        f"Output: {OUTPUT_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    generate_candidates()