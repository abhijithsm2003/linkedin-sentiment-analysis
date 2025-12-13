import pandas as pd
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path

# =========================
# NLTK SETUP (REQUIRED)
# =========================
nltk.download("vader_lexicon")

# =========================
# CONFIGURATION
# =========================
BASE_DIR = Path(r"C:\Users\DELL\Desktop\linkedin")

POSTS_DATA_PATH = BASE_DIR / "data" / "processed"
COMMENTS_DATA_PATH = BASE_DIR / "data" / "processed" / "cmts"

POSTS_RESULTS_PATH = BASE_DIR / "data" / "result"
COMMENTS_RESULTS_PATH = BASE_DIR / "data" / "result" / "cmts"

POSTS_RESULTS_PATH.mkdir(parents=True, exist_ok=True)
COMMENTS_RESULTS_PATH.mkdir(parents=True, exist_ok=True)

# =========================
# SENTIMENT ANALYZER
# =========================
class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        text = re.sub(r"#", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return " ".join(text.split())

    def analyze_dataframe(self, df, text_column, processed_col):
        df = df.copy()
        df[processed_col] = df[text_column].apply(self.clean_text)

        scores = df[processed_col].apply(self.sia.polarity_scores)
        df["sentiment_negative"] = scores.apply(lambda x: x["neg"])
        df["sentiment_neutral"] = scores.apply(lambda x: x["neu"])
        df["sentiment_positive"] = scores.apply(lambda x: x["pos"])
        df["sentiment_compound"] = scores.apply(lambda x: x["compound"])

        df["sentiment_label"] = df["sentiment_compound"].apply(
            lambda x: "Positive" if x >= 0.05 else "Negative" if x <= -0.05 else "Neutral"
        )
        return df

# =========================
# PROCESS POSTS
# =========================
def process_posts(analyzer):
    print("\nProcessing POSTS...")
    csv_files = list(POSTS_DATA_PATH.glob("*.csv"))

    if not csv_files:
        print("❌ No post CSV files found")
        return

    for file in csv_files:
        print(f"→ {file.name}")
        df = pd.read_csv(file)

        if "text" not in df.columns:
            print("  ⚠ 'text' column not found, skipping")
            continue

        df = analyzer.analyze_dataframe(df, "text", "processed_text")
        out = POSTS_RESULTS_PATH / f"results_{file.stem}.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"  ✓ Saved: {out}")

# =========================
# PROCESS COMMENTS
# =========================
def process_comments(analyzer):
    print("\nProcessing COMMENTS...")
    csv_files = list(COMMENTS_DATA_PATH.glob("*.csv"))

    if not csv_files:
        print("⚠ No comment CSV files found")
        return

    for file in csv_files:
        print(f"→ {file.name}")
        df = pd.read_csv(file)

        if "Comment" not in df.columns:
            print("  ⚠ 'Comment' column not found, skipping")
            continue

        df = analyzer.analyze_dataframe(df, "Comment", "processed_comment")
        out = COMMENTS_RESULTS_PATH / f"results_{file.stem}.csv"
        df.to_csv(out, index=False, encoding="utf-8")
        print(f"  ✓ Saved: {out}")

# =========================
# MAIN
# =========================
def main():
    print("=" * 60)
    print("LinkedIn Sentiment Analysis STARTED")
    print("=" * 60)

    analyzer = SentimentAnalyzer()
    process_posts(analyzer)
    process_comments(analyzer)

    print("\n✔ Sentiment Analysis COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
