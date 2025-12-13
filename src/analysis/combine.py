import pandas as pd
import json
import re
from pathlib import Path

# =====================================================
# CONFIGURATION (FIXED PATHS — THIS WAS THE PROBLEM)
# =====================================================
BASE_DIR = Path(r"C:\Users\DELL\Desktop\linkedin")

POSTS_RESULTS_PATH = BASE_DIR / "data" / "result"
COMMENTS_RESULTS_PATH = BASE_DIR / "data" / "result" / "cmts"

OUTPUT_CSV_PATH = BASE_DIR / "data" / "result" / "combined_posts_comments_long.csv"
OUTPUT_JSON_PATH = BASE_DIR / "data" / "result" / "combined_posts_comments_nested.json"


# =====================================================
# HELPERS
# =====================================================
def extract_urn_from_filename(filename):
    match = re.search(r'linkedin-(\d+)', filename)
    return match.group(1) if match else None


def extract_urn_id(urn_string):
    if pd.isna(urn_string):
        return None
    match = re.search(r'(\d+)$', str(urn_string))
    return match.group(1) if match else None


# =====================================================
# LOAD POSTS
# =====================================================
def load_posts_data():
    print("=" * 60)
    print("Loading Posts Data")
    print("=" * 60)

    posts_files = list(POSTS_RESULTS_PATH.glob("results_*.csv"))

    if not posts_files:
        print(f"No results files found in: {POSTS_RESULTS_PATH}")
        return None

    all_posts = []
    for file in posts_files:
        df = pd.read_csv(file, encoding="utf-8")
        all_posts.append(df)
        print(f"✓ Loaded {file.name} ({len(df)} rows)")

    combined = pd.concat(all_posts, ignore_index=True)
    print(f"Total posts loaded: {len(combined)}")
    return combined


# =====================================================
# LOAD COMMENTS
# =====================================================
def load_comments_data():
    print("\n" + "=" * 60)
    print("Loading Comments Data")
    print("=" * 60)

    if not COMMENTS_RESULTS_PATH.exists():
        print(f"Comments folder not found: {COMMENTS_RESULTS_PATH}")
        return None

    comment_files = list(COMMENTS_RESULTS_PATH.glob("results_*.csv"))

    if not comment_files:
        print(f"No comment files found in: {COMMENTS_RESULTS_PATH}")
        return None

    all_comments = []
    for file in comment_files:
        df = pd.read_csv(file, encoding="utf-8")
        urn_id = extract_urn_from_filename(file.name)

        if urn_id:
            df["post_urn_id"] = urn_id
            all_comments.append(df)
            print(f"✓ Loaded {file.name} ({len(df)} comments)")

    combined = pd.concat(all_comments, ignore_index=True)
    print(f"Total comments loaded: {len(combined)}")
    return combined


# =====================================================
# CREATE LONG CSV
# =====================================================
def create_long_format_csv(posts_df, comments_df):
    posts_df["post_urn_id"] = posts_df["full_urn"].apply(extract_urn_id)

    posts_df = posts_df.add_prefix("post_")
    posts_df.rename(columns={"post_post_urn_id": "post_urn_id"}, inplace=True)

    if comments_df is None:
        posts_df["has_comment"] = False
        return posts_df

    comments_df = comments_df.add_prefix("comment_")
    comments_df.rename(columns={"comment_post_urn_id": "post_urn_id"}, inplace=True)

    combined = pd.merge(posts_df, comments_df, on="post_urn_id", how="left")
    combined["has_comment"] = combined["comment_Comment"].notna()
    return combined


# =====================================================
# CREATE NESTED JSON
# =====================================================
def create_nested_json(posts_df, comments_df):
    posts_df["post_urn_id"] = posts_df["full_urn"].apply(extract_urn_id)

    grouped_comments = {}
    if comments_df is not None:
        for urn in comments_df["post_urn_id"].unique():
            grouped_comments[urn] = (
                comments_df[comments_df["post_urn_id"] == urn]
                .drop(columns=["post_urn_id"])
                .to_dict(orient="records")
            )

    nested = []
    for _, row in posts_df.iterrows():
        urn = row["post_urn_id"]
        comments = grouped_comments.get(urn, [])

        record = row.to_dict()
        record["comment_stats"] = {
            "total_comments": len(comments),
            "avg_sentiment_compound": (
                sum(c.get("sentiment_compound", 0) for c in comments) / len(comments)
                if comments else 0
            ),
        }
        record["comments"] = comments
        nested.append(record)

    return nested


# =====================================================
# MAIN
# =====================================================
def main():
    print("=" * 60)
    print("Hybrid Format: Long CSV + Nested JSON")
    print("=" * 60)

    posts_df = load_posts_data()
    if posts_df is None:
        print("❌ No posts data found")
        return

    comments_df = load_comments_data()

    csv_df = create_long_format_csv(posts_df, comments_df)
    csv_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"\n✓ CSV saved → {OUTPUT_CSV_PATH}")

    json_data = create_nested_json(posts_df, comments_df)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"✓ JSON saved → {OUTPUT_JSON_PATH}")
    print("\nDONE ✔")


if __name__ == "__main__":
    main()
