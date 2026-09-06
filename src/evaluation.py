# -*- coding: utf-8 -*-
"""
Classification evaluation suite.

Standard classification metrics for the cloud-classification benchmarks:
- Per-class precision, recall (sensitivity), and F1-score
- Overall accuracy and balanced accuracy (macro recall)
- Macro-averaged and weighted F1-scores
- Non-parametric percentile bootstrap confidence intervals (95% CI, B=1000)
- Confusion matrix generation
- High-resolution publication figure generation (300 DPI)
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, f1_score


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Calculates non-parametric percentile bootstrap confidence intervals.

    Returns mean, lower bound, and upper bound for Accuracy, Balanced Accuracy, and Macro-F1.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_accs = []
    boot_bal_accs = []
    boot_macro_f1s = []

    alpha = (1.0 - confidence_level) / 2.0
    lower_pct = alpha * 100.0
    upper_pct = (1.0 - alpha) * 100.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_bootstraps):
            boot_idx = rng.randint(0, n_samples, size=n_samples)
            b_true = y_true[boot_idx]
            b_pred = y_pred[boot_idx]

            b_acc = (b_true == b_pred).mean() * 100.0
            b_bal_acc = balanced_accuracy_score(b_true, b_pred) * 100.0
            b_f1 = f1_score(b_true, b_pred, average="macro", zero_division=0) * 100.0

            boot_accs.append(b_acc)
            boot_bal_accs.append(b_bal_acc)
            boot_macro_f1s.append(b_f1)

    boot_accs = np.array(boot_accs)
    boot_bal_accs = np.array(boot_bal_accs)
    boot_macro_f1s = np.array(boot_macro_f1s)

    return {
        "accuracy": {
            "mean": round(float(np.mean(boot_accs)), 2),
            "ci_lower": round(float(np.percentile(boot_accs, lower_pct)), 2),
            "ci_upper": round(float(np.percentile(boot_accs, upper_pct)), 2),
        },
        "balanced_accuracy": {
            "mean": round(float(np.mean(boot_bal_accs)), 2),
            "ci_lower": round(float(np.percentile(boot_bal_accs, lower_pct)), 2),
            "ci_upper": round(float(np.percentile(boot_bal_accs, upper_pct)), 2),
        },
        "macro_f1": {
            "mean": round(float(np.mean(boot_macro_f1s)), 2),
            "ci_lower": round(float(np.percentile(boot_macro_f1s, lower_pct)), 2),
            "ci_upper": round(float(np.percentile(boot_macro_f1s, upper_pct)), 2),
        },
    }


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    dataset_name: str,
    model_name: str,
    output_dir: Path = Path("artifacts/figures"),
    title_suffix: str = "",
) -> dict:
    """Generates classification metrics and a confusion matrix figure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    overall_acc = report_dict["accuracy"] * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = report_dict["macro avg"]["f1-score"] * 100.0
    weighted_f1 = report_dict["weighted avg"]["f1-score"] * 100.0

    # Non-parametric Bootstrap Confidence Intervals
    boot_ci = compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=1000, confidence_level=0.95, seed=42)

    # 1. Per-class metrics table
    rows = []
    for idx, cname in enumerate(class_names):
        prec = report_dict[cname]["precision"] * 100.0
        rec = report_dict[cname]["recall"] * 100.0
        f1 = report_dict[cname]["f1-score"] * 100.0
        sup = int(report_dict[cname]["support"])
        rows.append({
            "Cloud Class": cname,
            "Precision (%)": round(prec, 2),
            "Recall / Sensitivity (%)": round(rec, 2),
            "F1-Score (%)": round(f1, 2),
            "Support (Images)": sup,
        })

    df_report = pd.DataFrame(rows)

    # 2. Render Publication-Grade Confusion Matrix Heatmap
    fig_w = max(7.0, len(class_names) * 0.85)
    fig_h = max(5.5, len(class_names) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    title_str = f"Confusion Matrix: {dataset_name}"
    if title_suffix:
        title_str += f" ({title_suffix})"
    else:
        title_str += f" ({model_name.upper()})"

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title_str,
        ylabel="True Clouds",
        xlabel="Predicted Clouds",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm_norm.max() / 2.0
    fs = 8 if len(class_names) > 6 else 9
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=fs,
            )

    fig.tight_layout()
    sanitized_dataset = dataset_name.lower().replace(' ', '_').replace('-', '_')
    sanitized_model = model_name.lower().replace(' ', '_').replace('-', '_')
    fig_path = output_dir / f"confusion_matrix_{sanitized_dataset}_{sanitized_model}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    # Save Markdown Table & JSON Report
    md_table = df_report.to_markdown(index=False)
    report_file = output_dir / f"evaluation_metrics_{sanitized_dataset}_{sanitized_model}.json"
    metrics_summary = {
        "dataset": dataset_name,
        "model": model_name,
        "overall_accuracy": round(overall_acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "bootstrap_95ci": boot_ci,
        "per_class": rows,
        "figure_path": str(fig_path),
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return {
        "df_report": df_report,
        "md_table": md_table,
        "figure_path": str(fig_path),
        "overall_accuracy": round(overall_acc, 2),
        "balanced_accuracy": round(bal_acc, 2),
        "macro_f1": round(macro_f1, 2),
        "weighted_f1": round(weighted_f1, 2),
        "raw_overall_accuracy": float(overall_acc),
        "raw_balanced_accuracy": float(bal_acc),
        "raw_macro_f1": float(macro_f1),
        "bootstrap_95ci": boot_ci,
        "metrics_summary": metrics_summary,
    }
