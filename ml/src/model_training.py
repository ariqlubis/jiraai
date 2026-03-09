from scipy.sparse import random
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, accuracy_score
)

import matplotlib.pyplot as plt 
import seaborn as sns
from loguru import logger

PROCESSED_DATA_PATH = Path("ml/data/processed")
MODELS_PATH = Path("ml/models")
MODELS_PATH.mkdir(parents=True, exist_ok=True)

def load_splits():
    train = pd.read_csv(PROCESSED_DATA_PATH / "train.csv")
    # train = train.sample(n=150_000, random_state=42)
    val = pd.read_csv(PROCESSED_DATA_PATH / "val.csv")
    test = pd.read_csv(PROCESSED_DATA_PATH / "test.csv")
    logger.info(f"Loaded - train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test

def build_pipeline(target: str) -> Pipeline:
    logger.info(f"Building pipeline for target: '{target}'")
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=150_000,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
            analyzer="word"
        )),
        ("clf", SGDClassifier(
            class_weight="balanced",
            loss='modified_huber',
            random_state=42
        ))
    ])

def train_model(train: pd.DataFrame, target: str) -> Pipeline:
    df = train.dropna(subset=[target, "text"])
    logger.info(f"Training `{target}` on {len(df):,} rows")

    pipeline = build_pipeline(target)
    pipeline.fit(df['text'], df[target])
    logger.info("Training DONE.")
    return pipeline

def evaluate(pipeline: Pipeline, df: pd.DataFrame, target: str, split_name: str):
    df = df.dropna(subset=[target, "text"])
    y_true = df[target]
    y_pred = pipeline.predict(df['text'])

    f1 = f1_score(y_true, y_pred, average='weighted')
    acc = accuracy_score(y_true, y_pred)

    logger.info(f"\n── {target} | {split_name} ──")
    logger.info(f"Weighted F1: {f1:.4f} | Accuracy: {acc:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred)}")

    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"{target} — {split_name} Confusion Matrix (F1={f1:.3f})")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    plot_path = MODELS_PATH / f"confusion_{target}_{split_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {plot_path}")

    return {"f1": f1, "accuracy": acc}   

def save_model(pipeline: Pipeline, name: str):
    path = MODELS_PATH / f"{name}.joblib"
    joblib.dump(pipeline, path)
    logger.info(f"Model saved: {path}")
    return path


def run_training():
    logger.info("=" * 50)
    logger.info("JiraFlow AI — Training Pipeline START")
    logger.info("=" * 50)

    train, val, test = load_splits()
    results = {}

    # ── Model 1: Ticket Type Classifier ──────────────────────────────────────
    type_pipeline = train_model(train, target="label")
    results["type_val"]  = evaluate(type_pipeline, val,  "label", "val")
    results["type_test"] = evaluate(type_pipeline, test, "label", "test")
    save_model(type_pipeline, "type_classifier")

    # ── Model 2: Priority Classifier ─────────────────────────────────────────
    prio_pipeline = train_model(train, target="priority_label")
    results["prio_val"]  = evaluate(prio_pipeline, val,  "priority_label", "val")
    results["prio_test"] = evaluate(prio_pipeline, test, "priority_label", "test")
    save_model(prio_pipeline, "priority_classifier")

    # SUMMARY
    logger.info("\n" + "=" * 50)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 50)
    for key, metrics in results.items():
        logger.info(f"  {key:<20} F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}")

    type_f1 = results["type_test"]["f1"]
    prio_f1 = results["prio_test"]["f1"]

    logger.info("\n── Target Check ──")
    logger.info(f"  Type classifier F1:     {type_f1:.4f}  {'✅ >0.90' if type_f1 > 0.90 else '⚠️  below target, needs tuning'}")
    logger.info(f"  Priority classifier F1: {prio_f1:.4f}  {'✅ >0.85' if prio_f1 > 0.85 else '⚠️  below target, needs tuning'}")
    logger.info("=" * 50)

    return type_pipeline, prio_pipeline, results


if __name__ == "__main__":
    run_training()