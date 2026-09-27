import os
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_TRAIN = os.path.join(BASE_DIR, "data", "train")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

GROUND_TRUTH_PATH = os.path.join(
    DATA_TRAIN,
    "train_ground_truth.tsv"
)

CANDIDATE_PATH = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

def load_data():
    print("Loading ground truth...")

    if not os.path.exists(GROUND_TRUTH_PATH):
        raise FileNotFoundError(
            f"Ground truth not found: {GROUND_TRUTH_PATH}"
        )

    if not os.path.exists(CANDIDATE_PATH):
        raise FileNotFoundError(
            f"Candidate pairs not found: {CANDIDATE_PATH}\n"
            "Run 02_blocking.py first."
        )

    ground_truth = pd.read_csv(
        GROUND_TRUTH_PATH,
        sep="\t",
        dtype=str
    ).fillna("")

    candidates = pd.read_csv(
        CANDIDATE_PATH,
        sep="\t",
        dtype=str
    ).fillna("")

    return ground_truth, candidates


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def parse_ids(value):
    if not value or pd.isna(value):
        return set()

    return {
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    }


# ---------------------------------------------------------
# Evaluate blocking
# ---------------------------------------------------------

def evaluate_blocking():

    ground_truth, candidates = load_data()

    # Make lookup dictionary:
    # S1 ID -> true matching IDs
    truth_map = {}

    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]

        truth_map[s1_id] = parse_ids(
            row["matched_entity_ids"]
        )

    # Make candidate lookup:
    # S1 ID -> generated candidate IDs
    candidate_map = {}

    for _, row in candidates.iterrows():
        s1_id = row["source1_entity_id"]

        candidate_map[s1_id] = parse_ids(
            row["candidate_entity_ids"]
        )

    total_true_matches = 0
    recovered_true_matches = 0

    entities_with_matches = 0
    entities_with_all_matches_recovered = 0

    singleton_count = 0
    singleton_with_candidates = 0

    candidate_counts = []

    missed_examples = []

    # -----------------------------------------------------
    # Evaluate every Source 1 entity
    # -----------------------------------------------------

    for s1_id, true_ids in truth_map.items():

        candidate_ids = candidate_map.get(
            s1_id,
            set()
        )

        candidate_counts.append(
            len(candidate_ids)
        )

        # Singleton = no true matches
        if len(true_ids) == 0:

            singleton_count += 1

            if len(candidate_ids) > 0:
                singleton_with_candidates += 1

            continue

        entities_with_matches += 1

        total_true_matches += len(true_ids)

        recovered = true_ids.intersection(
            candidate_ids
        )

        recovered_true_matches += len(recovered)

        if recovered == true_ids:
            entities_with_all_matches_recovered += 1
        else:
            missed = true_ids - candidate_ids

            if len(missed_examples) < 20:
                missed_examples.append({
                    "source1_entity_id": s1_id,
                    "true_matches": ",".join(sorted(true_ids)),
                    "candidates": ",".join(sorted(candidate_ids)),
                    "missed": ",".join(sorted(missed))
                })

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    if total_true_matches > 0:
        candidate_recall = (
            recovered_true_matches /
            total_true_matches
        )
    else:
        candidate_recall = 0.0

    if entities_with_matches > 0:
        entity_full_recall = (
            entities_with_all_matches_recovered /
            entities_with_matches
        )
    else:
        entity_full_recall = 0.0

    average_candidates = (
        sum(candidate_counts) /
        len(candidate_counts)
        if candidate_counts
        else 0
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("BLOCKING EVALUATION")
    print("=" * 60)

    print(f"Total Source 1 entities: "
          f"{len(truth_map):,}")

    print(f"Entities with true matches: "
          f"{entities_with_matches:,}")

    print(f"Singleton entities: "
          f"{singleton_count:,}")

    print(f"Total true matches: "
          f"{total_true_matches:,}")

    print(f"True matches recovered: "
          f"{recovered_true_matches:,}")

    print(
        f"Candidate recall: "
        f"{candidate_recall:.4%}"
    )

    print(
        f"Entities with ALL matches recovered: "
        f"{entities_with_all_matches_recovered:,}"
    )

    print(
        f"Entity-level full recall: "
        f"{entity_full_recall:.4%}"
    )

    print(
        f"Average candidates per S1: "
        f"{average_candidates:.2f}"
    )

    print(
        f"Singletons receiving candidates: "
        f"{singleton_with_candidates:,}"
    )

    print("=" * 60)

    # -----------------------------------------------------
    # Show missed matches
    # -----------------------------------------------------

    if missed_examples:

        print("\nExamples of missed true matches:")
        print("-" * 60)

        for example in missed_examples:

            print(
                f"\nS1: {example['source1_entity_id']}"
            )

            print(
                f"True: {example['true_matches']}"
            )

            print(
                f"Missed: {example['missed']}"
            )

            print(
                f"Candidates: {example['candidates']}"
            )

    else:

        print(
            "\nNo true matches were missed!"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    evaluate_blocking()