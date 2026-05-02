import os
import argparse
import random
import time
import math
import glob
import csv
from collections import Counter, defaultdict

import torch
import psycopg2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from model import TreeCNN
from featurizer import parse_plan_json, PlanNode, PG_OPERATORS
from bandit import (ThompsonSamplingBandit, BAO_HINT_SETS)

DB_CONFIG = {
    "dbname": "bao",
    "user": "bao",
    "password": "bao",
    "host": "localhost",
    "port": 5432
}

STATEMENT_TIMEOUT_MS = 20000.0

# --- Configuration & Setup ---

class ExperienceReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        
    def add(self, query_id, hint_set_idx, plan_node, actual_time_ms):
        """
        Store an execution experience.
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            
        self.buffer.append({
            'query_id': query_id,
            'hint_set_idx': hint_set_idx,
            'plan_node': plan_node,
            'actual_log_time': math.log(max(actual_time_ms, 1.0))
        })

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

class TrainingMetrics:
    def __init__(self):
        self.query_records = []
        self.epoch_records = []
        self.training_losses = []

    @staticmethod
    def _percentile(values, percentile):
        if not values:
            return float("nan")

        sorted_values = sorted(values)
        idx = int(math.ceil((percentile / 100.0) * len(sorted_values))) - 1
        idx = max(0, min(idx, len(sorted_values) - 1))
        return sorted_values[idx]

    @staticmethod
    def _q_error(predicted_ms, actual_ms):
        predicted_ms = max(float(predicted_ms), 1.0)
        actual_ms = max(float(actual_ms), 1.0)
        return max(predicted_ms / actual_ms, actual_ms / predicted_ms)

    def add_query_result(
        self,
        epoch,
        query_name,
        query_id,
        selected_hint,
        predicted_log_time,
        actual_time_ms,
        timed_out,
        valid_plan_count
    ):
        predicted_time_ms = math.exp(float(predicted_log_time))
        q_error = self._q_error(predicted_time_ms, actual_time_ms)

        self.query_records.append({
            "epoch": epoch,
            "query_name": query_name,
            "query_id": query_id,
            "selected_hint": selected_hint,
            "predicted_log_time": float(predicted_log_time),
            "predicted_time_ms": predicted_time_ms,
            "actual_time_ms": actual_time_ms,
            "q_error": q_error,
            "timed_out": int(timed_out),
            "valid_plan_count": valid_plan_count,
        })

    def add_training_loss(self, epoch, loss):
        self.training_losses.append({
            "epoch": epoch,
            "training_loss": loss,
        })

    def summarize_epoch(self, epoch):
        records = [r for r in self.query_records if r["epoch"] == epoch]
        if not records:
            print(f"[*] Epoch {epoch} Metrics: no query records collected.")
            return

        latencies = [r["actual_time_ms"] for r in records]
        q_errors = [r["q_error"] for r in records]
        timeout_count = sum(r["timed_out"] for r in records)
        hint_counts = Counter(r["selected_hint"] for r in records)

        mean_latency = sum(latencies) / len(latencies)
        median_latency = self._percentile(latencies, 50)
        p95_latency = self._percentile(latencies, 95)
        max_latency = max(latencies)
        timeout_rate = timeout_count / len(records)

        median_q_error = self._percentile(q_errors, 50)
        p95_q_error = self._percentile(q_errors, 95)
        max_q_error = max(q_errors)

        epoch_summary = {
            "epoch": epoch,
            "query_count": len(records),
            "mean_latency_ms": mean_latency,
            "median_latency_ms": median_latency,
            "p95_latency_ms": p95_latency,
            "max_latency_ms": max_latency,
            "timeout_rate": timeout_rate,
            "median_q_error": median_q_error,
            "p95_q_error": p95_q_error,
            "max_q_error": max_q_error,
        }
        self.epoch_records.append(epoch_summary)

        print(f"[*] Epoch {epoch} Metrics")
        print(f"    Queries Evaluated:     {len(records)}")
        print(f"    Mean Latency:          {mean_latency:.2f} ms")
        print(f"    Median Latency:        {median_latency:.2f} ms")
        print(f"    P95 Latency:           {p95_latency:.2f} ms")
        print(f"    Max Latency:           {max_latency:.2f} ms")
        print(f"    Timeout Rate:          {timeout_rate:.2%}")
        print(f"    Median Q-Error:        {median_q_error:.3f}")
        print(f"    P95 Q-Error:           {p95_q_error:.3f}")
        print(f"    Max Q-Error:           {max_q_error:.3f}")
        print(f"    Hint Distribution:     {dict(sorted(hint_counts.items()))}")

    def print_final_summary(self):
        if not self.query_records:
            print("[*] Final Metrics: no query records collected.")
            return

        latencies = [r["actual_time_ms"] for r in self.query_records]
        q_errors = [r["q_error"] for r in self.query_records]
        timeout_count = sum(r["timed_out"] for r in self.query_records)
        hint_counts = Counter(r["selected_hint"] for r in self.query_records)

        print("\n[*] Final Metrics Summary")
        print(f"    Total Query Executions: {len(self.query_records)}")
        print(f"    Mean Latency:           {sum(latencies) / len(latencies):.2f} ms")
        print(f"    Median Latency:         {self._percentile(latencies, 50):.2f} ms")
        print(f"    P95 Latency:            {self._percentile(latencies, 95):.2f} ms")
        print(f"    Max Latency:            {max(latencies):.2f} ms")
        print(f"    Timeout Rate:           {timeout_count / len(self.query_records):.2%}")
        print(f"    Median Q-Error:         {self._percentile(q_errors, 50):.3f}")
        print(f"    P95 Q-Error:            {self._percentile(q_errors, 95):.3f}")
        print(f"    Max Q-Error:            {max(q_errors):.3f}")
        print(f"    Hint Distribution:      {dict(sorted(hint_counts.items()))}")

    def save_csvs(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        query_metrics_path = os.path.join(output_dir, "query_metrics.csv")
        with open(query_metrics_path, "w", newline="") as f:
            fieldnames = [
                "epoch",
                "query_name",
                "query_id",
                "selected_hint",
                "predicted_log_time",
                "predicted_time_ms",
                "actual_time_ms",
                "q_error",
                "timed_out",
                "valid_plan_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.query_records)

        epoch_metrics_path = os.path.join(output_dir, "epoch_metrics.csv")
        with open(epoch_metrics_path, "w", newline="") as f:
            fieldnames = [
                "epoch",
                "query_count",
                "mean_latency_ms",
                "median_latency_ms",
                "p95_latency_ms",
                "max_latency_ms",
                "timeout_rate",
                "median_q_error",
                "p95_q_error",
                "max_q_error",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.epoch_records)

        training_loss_path = os.path.join(output_dir, "training_loss.csv")
        with open(training_loss_path, "w", newline="") as f:
            fieldnames = ["epoch", "training_loss"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.training_losses)

        print(f"[*] Metrics CSV files saved to {output_dir}")

    def save_plots(self, output_dir):
        if plt is None:
            print("[!] matplotlib is not installed. Skipping plots.")
            return

        os.makedirs(output_dir, exist_ok=True)

        if self.epoch_records:
            epochs = [r["epoch"] for r in self.epoch_records]

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [r["mean_latency_ms"] for r in self.epoch_records], marker="o", label="Mean")
            plt.plot(epochs, [r["median_latency_ms"] for r in self.epoch_records], marker="o", label="Median")
            plt.plot(epochs, [r["p95_latency_ms"] for r in self.epoch_records], marker="o", label="P95")
            plt.xlabel("Epoch")
            plt.ylabel("Latency (ms)")
            plt.title("Latency by Epoch")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "latency_by_epoch.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [r["timeout_rate"] for r in self.epoch_records], marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Timeout Rate")
            plt.title("Timeout Rate by Epoch")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "timeout_rate_by_epoch.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, [r["median_q_error"] for r in self.epoch_records], marker="o", label="Median")
            plt.plot(epochs, [r["p95_q_error"] for r in self.epoch_records], marker="o", label="P95")
            plt.xlabel("Epoch")
            plt.ylabel("Q-Error")
            plt.title("Prediction Q-Error by Epoch")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "q_error_by_epoch.png"))
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(
                [r["epoch"] for r in self.training_losses],
                [r["training_loss"] for r in self.training_losses],
                marker="o"
            )
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("Training Loss by Epoch")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "training_loss_by_epoch.png"))
            plt.close()

        if self.query_records:
            actual = [r["actual_time_ms"] for r in self.query_records]
            predicted = [r["predicted_time_ms"] for r in self.query_records]

            plt.figure(figsize=(8, 8))
            plt.scatter(actual, predicted, alpha=0.7)
            max_value = max(max(actual), max(predicted), 1.0)
            plt.plot([1.0, max_value], [1.0, max_value], linestyle="--", color="black", label="Perfect Prediction")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Actual Runtime (ms)")
            plt.ylabel("Predicted Runtime (ms)")
            plt.title("Predicted vs Actual Runtime")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "predicted_vs_actual_runtime.png"))
            plt.close()

            hint_counts = Counter(r["selected_hint"] for r in self.query_records)
            hints = sorted(hint_counts.keys())
            counts = [hint_counts[h] for h in hints]

            plt.figure(figsize=(10, 6))
            plt.bar([str(h) for h in hints], counts)
            plt.xlabel("Hint Set")
            plt.ylabel("Selection Count")
            plt.title("Hint Selection Distribution")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "hint_selection_distribution.png"))
            plt.close()

            q_errors = [r["q_error"] for r in self.query_records]
            plt.figure(figsize=(10, 6))
            plt.hist(q_errors, bins=30)
            plt.xlabel("Q-Error")
            plt.ylabel("Frequency")
            plt.title("Q-Error Distribution")
            plt.grid(True, axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "q_error_distribution.png"))
            plt.close()

        print(f"[*] Metrics plots saved to {output_dir}")

class TreeDataset(Dataset):
    def __init__(self, buffer_samples):
        self.samples = buffer_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def tree_collate_fn(batch):
    """
    Since trees vary in structure, standard batching into contiguous tensors is difficult.
    We return a list of PlanNodes and a tensor of targets.
    """
    plan_nodes = [item['plan_node'] for item in batch]
    targets = torch.tensor([[item['actual_log_time']] for item in batch], dtype=torch.float32)
    return plan_nodes, targets

# --- Dataset Loader Facility ---

def load_job_queries(limit, split_ratio=1.0, seed=42, mode="train"):
    """
    Load JOB queries and split them by template to ensure generalization.
    Templates are identified by the numeric prefix of the filename (e.g., 1a.sql -> Template 1).
    """
    import re
    from collections import defaultdict

    sql_files = glob.glob('job_queries/*.sql')
    sql_files = [f for f in sql_files if 'fkindexes' not in f and 'schema' not in f]
    sql_files.sort()

    # Group queries by template
    template_groups = defaultdict(list)
    for f in sql_files:
        name = os.path.basename(f)
        template_id = re.match(r'(\d+)', name).group(1)
        template_groups[template_id].append(f)

    unique_templates = sorted(list(template_groups.keys()), key=int)
    random.seed(seed)
    random.shuffle(unique_templates)

    split_idx = int(len(unique_templates) * split_ratio)

    if mode == "train":
        selected_templates = unique_templates[:split_idx]
    else:
        selected_templates = unique_templates[split_idx:]

    selected_files = []
    for t_id in selected_templates:
        selected_files.extend(template_groups[t_id])

    selected_files.sort()
    if limit:
        selected_files = selected_files[:limit]

    queries = []
    for sql_file in selected_files:
        query_name = os.path.basename(sql_file).replace('.sql', '')
        with open(sql_file, 'r') as f:
            sql = f.read().replace(';', '')
            queries.append({"name": query_name.upper(), "sql": sql})

    print(f"[*] Mode: {mode.upper()} | Templates: {len(selected_templates)} | Queries: {len(queries)}")
    return queries

# --- Main Training & Simulation Loop ---

def main():
    parser = argparse.ArgumentParser(description="Bao Learned Optimizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--limit", type=int, default=None, help="Limit total queries in the split")
    parser.add_argument("--split", type=float, default=0.8, help="Train/Test split ratio (templates)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting templates")
    parser.add_argument("--metrics-dir", type=str, default="metrics", help="Directory for metrics CSVs and plots")
    args = parser.parse_args()

    queries = load_job_queries(args.limit, split_ratio=args.split, seed=args.seed, mode="train")

    # Initialize Core Modules
    in_channels = len(PG_OPERATORS) + 4
    model = TreeCNN(in_channels=in_channels, out_channels=128)
    optimizer = optim.Adam(model.parameters(), lr=0.0001    )
    loss_fn = nn.MSELoss()

    bandit = ThompsonSamplingBandit(model=model, num_mc_samples=5)
    replay_buffer = ExperienceReplayBuffer(capacity=5000)
    metrics = TrainingMetrics()

    print(f"[*] Starting Native Bao Training Loop on {len(queries)} JOB Benchmark Queries...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[!] Critical Error: Could not connect to PostgreSQL. {e}")
        return

    required_tables = ["title", "char_name", "cast_info", "movie_info", "movie_companies"]
    with conn.cursor() as cur:
        missing_tables = []
        for table_name in required_tables:
            cur.execute("""
                        SELECT EXISTS (SELECT 1
                                       FROM information_schema.tables
                                       WHERE table_schema = 'public'
                                         AND table_name = %s);
                        """, (table_name,))
            exists = cur.fetchone()[0]
            if not exists:
                missing_tables.append(table_name)

        if missing_tables:
            print(f"[!] Missing required IMDB tables: {', '.join(missing_tables)}")
            print("[!] Load ../imdb/schematext.sql and CSV data before training.")
            conn.close()
            return

        with conn.cursor() as cur:
            for epoch in range(args.epochs):
                epoch_num = epoch + 1
                print(f"--- Epoch {epoch_num}/{args.epochs} ---")

                # 1. Routing Phase (Thompson Sampling / Arm Selection)
                for q_id, q in enumerate(queries):
                    # Formulate the EXPLAIN plans natively from Postgres for each arm
                    arm_plans = []
                    for hint_idx, hint_str in BAO_HINT_SETS.items():
                        explain_query = f"/*+ {hint_str} */ EXPLAIN (FORMAT JSON) {q['sql']}"
                        try:
                            cur.execute(explain_query)
                            plan_json = cur.fetchone()[0][0]
                            plan_node = parse_plan_json(plan_json)
                            arm_plans.append(plan_node)
                        except Exception as e:
                            # Ex: Some hints can make queries un-plannable depending on DB schema config
                            conn.rollback()
                            print(f"[!] Warning: Query {q['name']} produced an error for Hint {hint_idx}: {e}")
                            arm_plans.append(None)

                    # Select best arm via Bandit evaluation against the true structural trees
                    best_arm_idx, predicted_log_time = bandit.select_arm(arm_plans)

                    # Fetch chosen arm strings
                    best_hint_str = BAO_HINT_SETS[best_arm_idx]
                    optimal_plan_node = arm_plans[best_arm_idx]

                    if optimal_plan_node is None:
                        print(f"[!] Warning: Query {q['name']} produced no valid plans across all hints. Skipping.")
                        continue

                    # Setup timeouts safely in ms, max training cap of 20000ms limit
                    cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)};")
                    bao_sql = f"/*+ {best_hint_str} */ {q['sql']}"

                    cur.execute("DISCARD PLANS;")  # Always flush caches for accurate learning
                    start_time = time.time()
                    timed_out = False
                    try:
                        cur.execute(bao_sql)
                        actual_time = (time.time() - start_time) * 1000
                    except psycopg2.errors.QueryCanceled:
                        print(
                            f"    -> Query {q['name']} timed out using Hint {best_arm_idx}. Applying heavy cost penalty.")
                        conn.rollback()
                        cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)};")
                        actual_time = STATEMENT_TIMEOUT_MS  # Time threshold penalty
                        timed_out = True
                    except Exception as e:
                        print(f"    -> Query {q['name']} failed execution using Hint {best_arm_idx}: {e}")
                        conn.rollback()
                        cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)};")
                        actual_time = STATEMENT_TIMEOUT_MS  # Safety failure penalty
                        timed_out = True

                    # Store structural truth inside Replay Buffer
                    replay_buffer.add(q_id, best_arm_idx, optimal_plan_node, actual_time)

                    valid_plan_count = sum(1 for plan in arm_plans if plan is not None)
                    metrics.add_query_result(
                        epoch=epoch_num,
                        query_name=q["name"],
                        query_id=q_id,
                        selected_hint=best_arm_idx,
                        predicted_log_time=predicted_log_time,
                        actual_time_ms=actual_time,
                        timed_out=timed_out,
                        valid_plan_count=valid_plan_count,
                    )

                    print(
                        f"[*] Evaluated Query {q['name']:<4} | "
                        f"Hint Selected: {best_arm_idx} | "
                        f"Predicted: {math.exp(float(predicted_log_time)):7.2f} ms | "
                        f"Actual: {actual_time:7.2f} ms | "
                        f"Timed Out: {timed_out}"
                    )

                print(f"[*] Epoch Finished. Replay buffer size: {len(replay_buffer.buffer)}")

                # 2. Training Phase (SGD / Adam)
                # Standard structural learning on truth parameters
                if len(replay_buffer.buffer) >= min(len(queries), 32):
                    model.train()
                    # Grab latest samples / subset matching reality
                    batch_samples = replay_buffer.sample(32)
                    dataset = TreeDataset(batch_samples)
                    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=tree_collate_fn)

                    batch_loss_sum = 0
                    for plan_nodes, targets in dataloader:
                        optimizer.zero_grad()

                        preds = []
                        for node in plan_nodes:
                            pred = model(node)
                            preds.append(pred)
                        preds_tensor = torch.stack(preds)  # [batch_size, 1]

                        loss = loss_fn(preds_tensor, targets)
                        loss.backward()
                        optimizer.step()

                        batch_loss_sum += loss.item()

                    avg_training_loss = batch_loss_sum / len(dataloader)
                    metrics.add_training_loss(epoch_num, avg_training_loss)
                    print(f"[*] Training Loss: {avg_training_loss:.4f}")
                else:
                    print("[*] Training skipped: replay buffer does not contain enough samples yet.")

                metrics.summarize_epoch(epoch_num)

        conn.close()

        print("\n[*] Training Sequence Complete.")

        metrics.print_final_summary()
        metrics.save_csvs(args.metrics_dir)
        metrics.save_plots(args.metrics_dir)

        os.makedirs("models", exist_ok=True)
        save_path = "models/bao_imdb.pt"
        torch.save(model.state_dict(), save_path)
        print(f"[*] Native DB Training Weights secured in {save_path}")

if __name__ == "__main__":
    main()
