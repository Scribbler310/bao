import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. DATA EXTRACTION FUNCTIONS
# ==========================================

def load_continuous_training_data(filepath):
    if not os.path.exists(filepath): return None, None
    pg_times, bao_times = [], []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle potential case sensitivity in CSV headers
            pg_val = row.get('postgres_time_ms')
            bao_val = row.get('actual_time_ms')
            if pg_val and bao_val:
                pg_times.append(float(pg_val) / 1000.0)
                bao_times.append(float(bao_val) / 1000.0)
    return np.array(pg_times), np.array(bao_times)


def load_holdout_test_data(test_filepath, optimal_filepath):
    """
    Refactored for Figure 11: Merges Benchmark results with Optimal Baselines.
    """
    if not os.path.exists(test_filepath): return None, None, None

    # 1. Load optimal values into a lookup dict
    optimals = {}
    if os.path.exists(optimal_filepath):
        with open(optimal_filepath, 'r', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                name = row.get('query_name')
                opt_ms = row.get('optimal_time_ms')
                if name and opt_ms:
                    optimals[name.upper()] = float(opt_ms) / 1000.0

    query_names, bao_diffs, optimal_diffs = [], [], []
    with open(test_filepath, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = (row.get('query_name')).upper()
            pg_s = float(row.get('postgres_time_ms')) / 1000.0
            bao_s = float(row.get('bao_time_ms')) / 1000.0

            if name in optimals:
                query_names.append(name)
                bao_diffs.append(bao_s - pg_s)
                optimal_diffs.append(optimals[name] - pg_s)

    return np.array(query_names), np.array(bao_diffs), np.array(optimal_diffs)


def extract_q_errors(filepath):
    if not os.path.exists(filepath): return None
    q_errors = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            val = row.get('q_error')
            if val: q_errors.append(float(val))
    return np.array(q_errors)

def extract_opt_vs_exec(metrics_dir):
    arms = [1, 5, 15, 25, 35, 45]
    opt_times, exec_times = [], []
    for arm in arms:
        filepath = os.path.join(metrics_dir, f"{arm}_arms_query_metrics.csv")
        if os.path.exists(filepath):
            tot_opt, tot_exec = 0.0, 0.0
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    opt_ms = row.get('optimization_time_ms') or 0
                    act_ms = row.get('actual_time_ms') or 0
                    tot_opt += float(opt_ms) / 1000.0 / 60.0
                    tot_exec += float(act_ms) / 1000.0 / 60.0
            opt_times.append(tot_opt)
            exec_times.append(tot_exec)
        else:
            opt_times.append(0)
            exec_times.append(0)
    return arms, opt_times, exec_times


# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def plot_figure_09(pg_times, bao_times, out_dir):
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
    plt.title("Figure 9 Replica: Percentile Latency")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(out_dir, "figure_09_percentile_latency.png"), dpi=300)
    plt.close()


def plot_figure_10(pg_times, bao_times, out_dir):
    print("[*] Generating Figure 10: Queries Finished Over Time...")
    pg_cumsum = np.cumsum(pg_times) / 3600.0
    bao_cumsum = np.cumsum(bao_times) / 3600.0
    plt.figure(figsize=(8, 6))
    plt.plot(bao_cumsum, np.arange(len(bao_times)), label='Bao', color='mediumblue')
    plt.plot(pg_cumsum, np.arange(len(pg_times)), label='PostgreSQL', color='orange')
    plt.xlabel("Time (hours)")
    plt.ylabel("Queries finished")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "figure_10_queries_over_time.png"), dpi=300)
    plt.close()


def plot_figure_11(query_names, bao_diffs, optimal_diffs, out_dir):
    print("[*] Generating Figure 11: Regression Analysis...")
    idx = np.argsort(bao_diffs)
    plt.figure(figsize=(15, 5))
    x = np.arange(len(query_names))
    plt.bar(x - 0.2, optimal_diffs[idx], 0.4, label='Optimal', color='forestgreen')
    plt.bar(x + 0.2, bao_diffs[idx], 0.4, label='Bao', color='mediumblue')
    plt.xticks(x, query_names[idx], rotation=90, fontsize=7)
    plt.ylabel("Diff from PostgreSQL (s)")
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_11_absolute_diff.png"), dpi=300)
    plt.close()


def plot_figure_12(arms, opt_times, exec_times, out_dir):
    if not any(opt_times): return
    print("[*] Generating Figure 12: Tradeoff Analysis...")
    plt.figure(figsize=(8, 6))
    plt.bar(arms, opt_times, width=4, label='Optimization', color='tab:blue')
    plt.bar(arms, exec_times, width=4, bottom=opt_times, label='Execution', color='tab:orange')
    plt.ylabel("Workload time (m)")
    plt.xlabel("Number of arms")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "figure_12_opt_vs_exec.png"), dpi=300)
    plt.close()


def plot_figure_15b(q_errors, out_dir):
    """
    Generates Figure 15b: Median Q-Error over time.
    Replicates the model convergence visualization from the Bao paper.
    """
    if q_errors is None or len(q_errors) == 0:
        print("[!] Skipping Figure 15b: No Q-Error data found.")
        return

    print("[*] Generating Figure 15b: Median Q-Error...")

    # The paper uses a rolling window to show the trend of the median.
    # A window of 100 matches the 'epoch' size used in your training loop.
    window_size = 100
    rolling_median = [
        np.median(q_errors[max(0, i - window_size):i + 1])
        for i in range(len(q_errors))
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(rolling_median, color='mediumblue', linewidth=1.5)

    # Formatting to match Figure 15b in the paper
    plt.yscale('log')  # Q-Error is typically viewed on a log scale
    plt.xlabel("Queries processed")
    plt.ylabel("Median Q-Error")
    plt.title("Figure 15b Replica: Model Convergence (Median Q-Error)")
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure_15b_median_q_error.png"), dpi=300)
    plt.close()


# ==========================================
# 3. MAIN
# ==========================================

def generate_analysis():
    out_dir = "paper_figures"
    os.makedirs(out_dir, exist_ok=True)
    m_dir = "metrics"

    # Original Figures (9, 10, 15b)
    pg, bao = load_continuous_training_data(os.path.join(m_dir, "query_metrics.csv"))
    if pg is not None:
        plot_figure_09(pg, bao, out_dir)
        plot_figure_10(pg, bao, out_dir)
        plot_figure_15b(extract_q_errors(os.path.join(m_dir, "query_metrics.csv")), out_dir)

    # Figure 11
    q, b_d, o_d = load_holdout_test_data(os.path.join(m_dir, "holdout_test_metrics.csv"),
                                         os.path.join(m_dir, "optimal_baselines.csv"))
    if q is not None: plot_figure_11(q, b_d, o_d, out_dir)

    # Figure 12
    arms, ot, et = extract_opt_vs_exec(m_dir)
    plot_figure_12(arms, ot, et, out_dir)


if __name__ == "__main__":
    generate_analysis()