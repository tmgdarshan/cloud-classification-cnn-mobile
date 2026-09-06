import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_JSON = REPO_ROOT / "artifacts" / "harmonized_results" / "harmonized_summary_resnet18.json"
REPORT_FIG_PATH = REPO_ROOT / "report" / "figures" / "transfer_asymmetry_recall_resnet18.png"
ARTIFACT_FIG_PATH = REPO_ROOT / "artifacts" / "figures" / "transfer_asymmetry_recall_resnet18.png"

def generate_transfer_asymmetry_plot():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    ccsn_to_gcd = data.get("cross_source_ccsn_to_gcd", data.get("zeroshot_ccsn_to_gcd"))["test_holdout"]["per_class"]
    gcd_to_ccsn = data.get("cross_source_gcd_to_ccsn", data.get("zeroshot_gcd_to_ccsn"))["test_holdout"]["per_class"]

    classes = ["cumulus", "altocumulus", "cirrus", "stratocumulus", "cumulonimbus"]
    class_labels = ["Cumulus\n($C_1$)", "Altocumulus\n($C_2$)", "Cirrus\n($C_3$)", "Stratocumulus\n($C_4$)", "Cumulonimbus\n($C_5$)"]

    recalls_ccsn_to_gcd = [item["Recall / Sensitivity (%)"] for item in ccsn_to_gcd]
    recalls_gcd_to_ccsn = [item["Recall / Sensitivity (%)"] for item in gcd_to_ccsn]

    x = np.arange(len(classes))
    width = 0.35

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    color_ccsn_to_gcd = "#1f77b4"  # Steel blue
    color_gcd_to_ccsn = "#d62728"  # Crimson

    rects1 = ax.bar(x - width/2, recalls_ccsn_to_gcd, width, label="CCSN $\\to$ GCD (Regional to Whole-Sky)", color=color_ccsn_to_gcd, alpha=0.9, edgecolor="black", linewidth=0.8)
    rects2 = ax.bar(x + width/2, recalls_gcd_to_ccsn, width, label="GCD $\\to$ CCSN (Whole-Sky to Regional)", color=color_gcd_to_ccsn, alpha=0.9, edgecolor="black", linewidth=0.8)

    ax.set_ylabel("Class-Wise Recall / Sensitivity (%)", fontsize=11, fontweight="bold")
    ax.set_title("Cross-Source Transfer Asymmetry by Compatibility Class (ResNet-18)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", fontsize=9.5, loc="upper right")

    # Add values on top of bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1f77b4")

    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#d62728")

    # Annotate the Cb directional disparity
    cb_diff = recalls_ccsn_to_gcd[4] - recalls_gcd_to_ccsn[4]
    ax.annotate(f"{recalls_ccsn_to_gcd[4]:.1f}% vs. {recalls_gcd_to_ccsn[4]:.1f}%\n(Directional Disparity)",
                xy=(x[4] + width/2, recalls_gcd_to_ccsn[4]),
                xytext=(x[4] - 0.1, 45),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
                fontsize=8.5, fontweight="bold", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#d62728", lw=1))

    fig.tight_layout()
    REPORT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORT_FIG_PATH, bbox_inches="tight")
    fig.savefig(ARTIFACT_FIG_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved transfer asymmetry figure to {REPORT_FIG_PATH} and {ARTIFACT_FIG_PATH}")

if __name__ == "__main__":
    generate_transfer_asymmetry_plot()
