import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. DATA EXTRACTION FUNCTIONS
# ==========================================

def load_continuous_training_data(filepath):
    """
    Extracts data for Figures 9 and 10, representing continuous online learning.
    Iterates through the CSV exactly once.
    """
    if not os.path.exists(filepath):
        print(f"[!] Warning: Training data {filepath} not found.")
        return None, None

    pg_times = []
    bao_times = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): v for k, v in raw_row.items() if k}
            if not row: continue

            pg_times.append(float(row['postgres_time_ms']) / 1000.0)
            bao_times.append(float(row['actual_time_ms']) / 1000.0)

    return np.array(pg_times), np.array(bao_times)


def load_holdout_test_data(filepath):
    """
    Extracts data for Figure 11, representing the strict query regression test.
    This expects a CSV with 'query_name', 'postgres_time_ms', 'bao_time_ms',
    and 'optimal_time_ms' for queries completely hidden during training.
    """
    if not os.path.exists(filepath):
        print(f"[!] Warning: Holdout test data {filepath} not found.")
        return None, None, None

    query_names = []
    bao_diffs = []
    optimal_diffs = []

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {k.strip(): v for k, v in raw_row.items() if k}
            if not row: continue

            pg_s = float(row['postgres_time_ms']) / 1000.0
            bao_s = float(row['bao_time_ms']) / 1000.0
            opt_s = float(row['optimal_time_ms']) / 1000.0

            query_names.append(row['query_name'])
            # Negative means faster than PostgreSQL
            bao_diffs.append(bao_s - pg_s)
            optimal_diffs.append(opt_s - pg_s)

    return np.array(query_names), np.array(bao_diffs), np.array(optimal_diffs)


# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def plot_figure_09(pg_times, bao_times, out_dir):
    """Generates Figure 9: Percentile Latency."""
    print("[*] Generating Figure 9: Percentile Latency...")
    percentiles = [50, 95, 99, 99.5]
    pg_p = np.percentile(pg_times, percentiles)
    bao_p = np.percentile(bao_times, percentiles)

    x = np.arange(len(percentiles))
    width = 0.35

    plt.figure(figsize=(8, 6))
    plt.bar(x - width / 2, bao_p, width, label='Bao', color='mediumblue')
    plt.bar(x + width / 2, pg_p, width, label='PostgreSQL', color='orange')
    plt.xticks(x, [f"{p}%" for p in percentiles])
    plt.ylabel("Wall time (s)")
    plt.xlabel("Percentile")
    plt.title("Figure 9 Replica: Percentile Latency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_09_percentile_latency.png"), dpi=300)
    plt.close()


def plot_figure_10(pg_times, bao_times, out_dir):
    """Generates Figure 10: Queries Finished Over Time."""
    print("[*] Generating Figure 10: Queries Finished Over Time...")
    pg_cumsum_hours = np.cumsum(pg_times) / 3600.0
    bao_cumsum_hours = np.cumsum(bao_times) / 3600.0
    y_queries = np.arange(1, len(pg_times) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(bao_cumsum_hours, y_queries, label='Bao', color='mediumblue', linewidth=2)
    plt.plot(pg_cumsum_hours, y_queries, label='PostgreSQL', color='orange', linewidth=2)
    plt.xlabel("Time (hours)")
    plt.ylabel("Queries finished")
    plt.title("Figure 10 Replica: Queries Finished Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_10_queries_over_time.png"), dpi=300)
    plt.close()


def plot_figure_11(query_names, bao_diffs, optimal_diffs, out_dir):
    """Generates Figure 11: Absolute Difference from PostgreSQL (with optimal baseline)."""
    print("[*] Generating Figure 11: Query Regression Analysis...")
    # Sort by Bao's difference
    sorted_indices = np.argsort(bao_diffs)
    sorted_bao = bao_diffs[sorted_indices]
    sorted_opt = optimal_diffs[sorted_indices]
    sorted_names = query_names[sorted_indices]

    plt.figure(figsize=(15, 5))
    x = np.arange(len(sorted_names))
    width = 0.4

    # Plot optimal baseline first (green), then Bao's performance
    plt.bar(x - width / 2, sorted_opt, width, label='Optimal', color='forestgreen')
    plt.bar(x + width / 2, sorted_bao, width, label='Bao', color='mediumblue')

    plt.xticks(x, sorted_names, rotation=90, fontsize=7)
    plt.ylabel("Difference from PostgreSQL (s)")
    plt.xlabel("Query ID")
    plt.title("Figure 11 Replica: Absolute Difference in Query Latency (Negative is Better)")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_11_absolute_diff.png"), dpi=300)
    plt.close()


# ==========================================
# 3. MAIN EXECUTION PIPELINE
# ==========================================

def generate_analysis():
    print("[*] Starting Bao result analysis pipeline...")
    out_dir = "paper_figures"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Process Continuous Training Figures (9 & 10)
    train_file = os.path.join("metrics", "query_metrics.csv")
    pg_times, bao_times = load_continuous_training_data(train_file)

    if pg_times is not None:
        plot_figure_09(pg_times, bao_times, out_dir)
        plot_figure_10(pg_times, bao_times, out_dir)

    # 2. Process Holdout Test Figures (11)
    # Note: You will need to generate this CSV by running a test script
    # that evaluates a holdout set without updating the model weights.
    test_file = os.path.join("metrics", "holdout_test_metrics.csv")
    query_names, bao_diffs, opt_diffs = load_holdout_test_data(test_file)

    if query_names is not None:
        plot_figure_11(query_names, bao_diffs, opt_diffs, out_dir)

    print("\n[*] Analysis complete! Check the paper_figures/ directory.")


if __name__ == "__main__":
    generate_analysis()