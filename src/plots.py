import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
files = {
    'ResNet18 (Baseline)': 'gcd_training_metrics.csv',
    'ResNet34 (Baseline)': 'gcd_training_metrics_resnet34.csv',
    'ResNet34 (Regularized)': 'gcd_training_metrics_resnet34_reg.csv'
}

# Colors and markers for distinction in black & white print
styles = {
    'ResNet18 (Baseline)': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},  # Blue
    'ResNet34 (Baseline)': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},  # Orange
    'ResNet34 (Regularized)': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'}  # Green
}


def plot_comparison():
    # Setup the figure
    plt.style.use('seaborn-v0_8-paper')  # Professional academic style
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loop through files and plot data
    for name, filename in files.items():
        try:
            df = pd.read_csv(filename)

            # Plot Accuracy
            ax1.plot(df['epoch'], df['val_acc'], label=name,
                     color=styles[name]['color'],
                     marker=styles[name]['marker'],
                     linestyle=styles[name]['linestyle'],
                     linewidth=1.5, markersize=4)

            # Plot Loss
            ax2.plot(df['epoch'], df['val_loss'], label=name,
                     color=styles[name]['color'],
                     marker=styles[name]['marker'],
                     linestyle=styles[name]['linestyle'],
                     linewidth=1.5, markersize=4)

        except FileNotFoundError:
            print(f"⚠️ Warning: Could not find {filename}")

    # --- Formatting Plot 1 (Accuracy) ---
    ax1.set_title('Validation Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Accuracy (%)', fontsize=10)
    ax1.legend(loc='lower right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Formatting Plot 2 (Loss) ---
    ax2.set_title('Validation Loss Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss (Cross-Entropy)', fontsize=10)
    ax2.legend(loc='upper left', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Save and Show
    plt.tight_layout()
    plt.savefig('model_comparison_paper.png', dpi=300)
    plt.savefig('model_comparison_paper.pdf')  # PDF is better for LaTeX/Papers
    print("✅ Plots saved as 'model_comparison_paper.png' and '.pdf'")
    plt.show()


if __name__ == "__main__":
    plot_comparison()