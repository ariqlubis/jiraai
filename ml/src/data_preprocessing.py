import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
import re

RAW_DATA_PATH = Path("ml/data/raw")
PROCESSED_DATA_PATH = Path("ml/data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)  

KEEP_COLS = ["id", "title", "description", "type", "priority", "project_name"]

labels_map = {
    "Bug": "Bug",
    "New Feature": "Feature",
    "Feature Request": "Feature",
    "Task": "Task",
    "Improvement": "Task",
    "Sub-task": "Task"
}

def load_data(filename: str = "jira_issues.csv") -> pd.DataFrame:
    logger.info(f"Loading data from {filename}")
    logger.info(f"Data shape: {pd.read_csv(RAW_DATA_PATH / filename).shape}")
    return pd.read_csv(RAW_DATA_PATH / filename)

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns relevant to our domain model."""
    available = [col for col in KEEP_COLS if col in df.columns]
    missing = set(KEEP_COLS) - set(available)
    if missing:
        logger.warning(f"Missing columns (will skip): {missing}")
    return df[available].copy()

def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Mapping labels")
    df["label"] = df["type"].map(labels_map)
    return df

def clean_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["description"])
    after = len(df)
    logger.info(f"Dropped {before - after} rows with null description")
    return df

def clean_priority(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["priority"].notna()]
    logger.info(f"Dropped {before - len(df)} rows with null priority")

    df['priority'] = df['priority'].str.strip().str.title()

    priority_map = {
        "Blocker":  "Critical",
        "Critical": "Critical",
        "Major":    "High",
        "Minor":    "Low",
        "Trivial":  "Low",
        "Optional": "Low",
    }
    df['priority_label'] = df['priority'].map(priority_map)

    df = df.dropna(subset=["priority_label"]).copy()
    logger.info(f"Priority distribution:\n{df['priority_label'].value_counts()}")
    return df

def build_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = "[TITLE ]" + df["title"] + " [DESC] " + df["description"]
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df['text'].str.len() > 20].copy()
    df["text"] = df["text"].apply(normalize_text)
    logger.info(f"Text feature built! Avg length: {df['text'].str.len().mean()} chars")
    return df

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\d+", "NUM", text) 
    return text

def handle_imb_report(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("\n-Label Distribution-")
    counts = df['label'].value_counts()
    pcts = df['label'].value_counts(normalize=True) * 100
    for label in counts.index:
        logger.info(f"{label}: {counts[label]} ({pcts[label]:.2f}%)")


def save_processed(df: pd.DataFrame) -> None:
    from sklearn.model_selection import train_test_split
    
    final_cols = ["id", 'text', 'label', 'priority_label', 'project_name']
    available_final = [c for c in final_cols if c in df.columns]
    df_final = df[available_final].copy()
    before = len(df_final)
    df_final = df_final.dropna(subset=["label", "priority_label"]).copy()
    if len(df_final) < before:
        logger.info(f"Safety net: dropped {before - len(df_final):,} rows with NaN labels")
    
    output_path = PROCESSED_DATA_PATH / "jira_issues_processed.csv"
    logger.info(f"Saved full dataset: {output_path} ({len(df_final):,} rows)")
    df_final.to_csv(output_path, index=False)

    train, temp = train_test_split(
        df_final,
        test_size=0.3,
        random_state=42,
        stratify=df_final["label"]
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=42,
        stratify=temp["label"]
    )

    train.to_csv(PROCESSED_DATA_PATH / "train.csv", index=False)
    val.to_csv(PROCESSED_DATA_PATH / "val.csv", index=False)
    test.to_csv(PROCESSED_DATA_PATH / "test.csv", index=False)

    logger.info(f"Split: train={len(train):,} | val={len(val):,} | test={len(test):,}")
    logger.info(f"Saved to: {PROCESSED_DATA_PATH}/")

def run_pipeline(filename: str = "jira_issues.csv") -> pd.DataFrame:
    logger.info("==== JiraAI - Preprocessing Pipeline START ====")

    df = load_data(filename)
    df = select_columns(df)
    df = map_labels(df)
    df = clean_nulls(df)
    df = clean_priority(df)
    df = build_text_feature(df)

    handle_imb_report(df)
    save_processed(df)

    logger.info(f"Pipeline DONE. Final dataset: {len(df):,} rows ready for training")

    return df


if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "jira_issues.csv"
    run_pipeline(filename)
