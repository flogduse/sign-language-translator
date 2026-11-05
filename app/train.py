"""Train a fingerspelling classifier on MediaPipe hand landmarks.

    python -m app.train --data data/asl_landmarks.npz

Produces:
    models/asl_letters.joblib  — the winning model (joblib)
    models/labels.json         — index → letter, the contract the runtime uses
    docs/eval/metrics.json     — full comparison + winning model's report
    docs/eval/confusion.png    — confusion matrix of the winning model

Design notes
------------
* Features are produced by `app.features.to_features`, the same function the
  real-time runtime uses — training and inference cannot drift apart.
* The 29-class Kaggle-derived dataset is normalized per-sample before the
  split (there is no train/test leak: each sample is a still image).
* Model choice is measured, not assumed: three cheap, diverse classifiers are
  trained and the comparison lands in `metrics.json` next to the winner.
* Class 26 ('del'), 27 ('nothing'), 28 ('space') are kept when well-populated;
  single-sample classes are dropped and reported so the count stays honest.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import N_FEATURES, to_features

LETTERS = [chr(ord("A") + i) for i in range(26)] + ["DELETE", "NOTHING", "SPACE"]
MIN_CLASS_SAMPLES = 5
RANDOM_SEED = 11


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    data = np.load(path)
    X_raw, y = data["landmarks"], data["labels"].astype(np.int64)
    labels, counts = np.unique(y, return_counts=True)
    dropped = {name(i): int(c) for i, c in zip(labels, counts) if c < MIN_CLASS_SAMPLES}
    keep = np.isin(y, labels[counts >= MIN_CLASS_SAMPLES])
    Xr, kept = X_raw[keep], y[keep]
    if dropped:
        print(f"dropped low-population classes: {dropped}")
    X = np.stack([to_features(x) for x in Xr])
    return X, kept, [name(i) for i in np.unique(kept)], []


def name(index: int) -> str:
    try:
        return LETTERS[index]
    except IndexError:
        return f"unknown-{index}"


def build_models() -> dict:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=10.0, max_iter=2000, n_jobs=-1, random_state=RANDOM_SEED),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_SEED
        ),
        "hist_gbc": HistGradientBoostingClassifier(
            max_iter=250, random_state=RANDOM_SEED
        ),
    }


def confusion_png(cm: np.ndarray, labels: list[str], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10), dpi=140)
    ax.imshow(cm, cmap="viridis")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test split)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/asl_landmarks.npz"))
    parser.add_argument("--out-model", type=Path, default=Path("models/asl_letters.joblib"))
    parser.add_argument("--out-labels", type=Path, default=Path("models/labels.json"))
    parser.add_argument("--out-metrics", type=Path, default=Path("docs/eval/metrics.json"))
    parser.add_argument("--out-confusion", type=Path, default=Path("docs/eval/confusion.png"))
    args = parser.parse_args()

    X, y, class_names, _ = load_dataset(args.data)
    print(f"samples: {len(y)}   classes: {len(class_names)}   features: {N_FEATURES}")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    print(f"train: {len(ytr)}   test: {len(yte)}")

    metrics: dict = {"random_seed": RANDOM_SEED, "samples": len(y), "classes": class_names, "models": {}}
    best_name, best_model, best_acc = None, None, -1.0

    for name_key, model in build_models().items():
        start = time.perf_counter()
        model.fit(Xtr, ytr)
        elapsed = time.perf_counter() - start
        probs = model.predict_proba(Xte)
        preds = model.classes_[probs.argmax(axis=1)] if hasattr(model, "classes_") else model.predict(Xte)
        acc = accuracy_score(yte, preds)
        report = classification_report(yte, preds, output_dict=True, zero_division=0)
        print(f"{name_key:14s} acc={acc:.4f} top3={top_k_accuracy_score(yte, probs, k=3):.4f} ({elapsed:.1f}s)")
        metrics["models"][name_key] = {
            "accuracy": round(acc, 4),
            "precision_macro": round(report["macro avg"]["precision"], 4),
            "recall_macro": round(report["macro avg"]["recall"], 4),
            "per_class": {
                label: {
                    "precision": round(v["precision"], 4),
                    "recall": round(v["recall"], 4),
                    "f1": round(v["f1-score"], 4),
                    "support": int(v["support"]),
                }
                for label, v in report.items()
                if label not in ("accuracy", "macro avg", "weighted avg")
            },
        }
        if acc > best_acc:
            best_name, best_model, best_acc = name_key, model, acc

    winner = best_model
    print(f"\nbest: {best_name} acc={best_acc:.4f}")

    classes = np.unique(y)
    def pretty(targets: np.ndarray) -> list[str]:
        # `winner` predicts integer class ids; str-ify the label for humans.
        return [name(int(t)) for t in targets]

    preds = winner.predict(Xte)
    cm = confusion_matrix([name(int(t)) for t in yte], [name(int(p)) for p in preds], labels=class_names)
    confusion_png(cm, class_names, out_absolute(args.out_confusion))

    out_absolute(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner, out_absolute(args.out_model))
    print(f"wrote {args.out_model}")

    classes_json = {str(i): classname for i, classname in enumerate(LETTERS)}
    out_absolute(args.out_labels).write_text(json.dumps(classes_json, indent=2), encoding="utf-8")
    print(f"wrote {args.out_labels}")

    out_absolute(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    metrics["winner"] = best_name
    metrics["winner_accuracy"] = round(best_acc, 4)
    out_absolute(args.out_metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {args.out_metrics}")
    return 0


def out_absolute(p: Path) -> Path:
    return Path(p).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
