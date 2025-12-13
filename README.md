# LinkedIn Post & Comment Sentiment Analysis

This project performs end-to-end sentiment analysis on LinkedIn posts and their comments.
It includes URL extraction, comment cleaning, post–comment merging, text preprocessing,
sentiment scoring using VADER, and extensive visual analytics.

## Pipeline Overview

extract_urls  
→ clean_comments  
→ merge_posts_and_comments  
→ json_to_csv  
→ text_cleaning  
→ sentiment_analysis  
→ visualization

## Project Structure

- `scr/analysis/` – text processing and sentiment analysis
- `scr/cleaning/` – cleaning raw LinkedIn comment files
- `scr/combine/` – merging posts with comments
- `scr/json_to_csv/` – flattening JSON to CSV
- `scr/post_url/` – extracting LinkedIn post URLs
- `scr/visualization/` – generating plots
- `scr/utils/` – shared helper utilities
- `data/` – raw, processed, plots, and result datasets
- `tests/` – unit tests for analysis, processing, and utilities
- `config/` – configuration file

## Key Outputs

- Cleaned post–comment dataset (`posts_with_cmnts_cleaned.csv`)
- Sentiment-scored dataset (`posts_with_cmnts_sentiment.csv`)
- 30+ analytical plots saved as PNG files

## Sentiment Method

- VADER Sentiment Analyzer
- Output fields:
  - `sentiment_compound`
  - `sentiment_label` (positive / neutral / negative)

## Author
Abhijith S M
