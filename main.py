import os
import argparse
import random
import time
import math
import glob
import csv
from collections import Counter

import torch
import psycopg2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from model import TreeCNN
from featurizer import parse_plan_json, PG_OPERATORS
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
    def __init__(self, capacity=2000):  # Changed to 2000 to match paper
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

    def sample_with_replacement(self):
        """
        Thompson Sampling requires bootstrapping.
        We draw |E| samples with replacement from the buffer.
        """
        n_samples = len(self.buffer)
        return [random.choice(self.buffer) for _ in range(n_samples)]


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
            self, cycle, query_name, query_id, selected_hint, predicted_log_time,
            actual_time_ms, postgres_time_ms, timed_out, valid_plan_count
    ):
        predicted_time_ms = math.exp(float(predicted_log_time))
        q_error = self._q_error(predicted_time_ms, actual_time_ms)

        self.query_records.append({
            "epoch": cycle,  # Used as cycle tracker now
            "query_name": query_name,
            "query_id": query_id,
            "selected_hint": selected_hint,
            "predicted_log_time": float(predicted_log_time),
            "predicted_time_ms": predicted_time_ms,
            "actual_time_ms": actual_time_ms,
            "postgres_time_ms": postgres_time_ms,
            "q_error": q_error,
            "timed_out": int(timed_out),
            "valid_plan_count": valid_plan_count,
        })

    def add_training_loss(self, cycle, loss):
        self.training_losses.append({
            "epoch": cycle,
            "training_loss": loss,
        })

    def summarize_cycle(self, cycle):
        records = [r for r in self.query_records if r["epoch"] == cycle]
        if not records:
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
            "epoch": cycle,
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

        print(f"[*] Cycle {cycle} Metrics (Last 100 queries)")
        print(f"    Mean Latency:          {mean_latency:.2f} ms")
        print(f"    P95 Latency:           {p95_latency:.2f} ms")
        print(f"    Timeout Rate:          {timeout_rate:.2%}")
        print(f"    Hint Distribution:     {dict(sorted(hint_counts.items()))}")

    def print_final_summary(self):
        if not self.query_records:
            return

        latencies = [r["actual_time_ms"] for r in self.query_records]
        q_errors = [r["q_error"] for r in self.query_records]
        timeout_count = sum(r["timed_out"] for r in self.query_records)

        print("\n[*] Final Metrics Summary")
        print(f"    Total Query Executions: {len(self.query_records)}")
        print(f"    Mean Latency:           {sum(latencies) / len(latencies):.2f} ms")
        print(f"    P95 Latency:            {self._percentile(latencies, 95):.2f} ms")
        print(f"    Timeout Rate:           {timeout_count / len(self.query_records):.2%}")
        print(f"    P95 Q-Error:            {self._percentile(q_errors, 95):.3f}")

    def save_csvs(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        query_metrics_path = os.path.join(output_dir, "query_metrics.csv")
        with open(query_metrics_path, "w", newline="") as f:
            fieldnames = [
                "epoch", "query_name", "query_id", "selected_hint", "predicted_log_time",
                "predicted_time_ms", "actual_time_ms", "postgres_time_ms", "q_error",
                "timed_out", "valid_plan_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.query_records)


class TreeDataset(Dataset):
    def __init__(self, buffer_samples):
        self.samples = buffer_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def tree_collate_fn(batch):
    plan_nodes = [item['plan_node'] for item in batch]
    targets = torch.tensor([[item['actual_log_time']] for item in batch], dtype=torch.float32)
    return plan_nodes, targets


# --- Dataset Loader Facility ---

def load_job_queries(limit, split_ratio=1.0, seed=42, mode="train"):
    """
    Load queries directly from the job_d folder to represent a continuous stream.
    """
    sql_files = glob.glob('job_d/*.sql')
    sql_files = [f for f in sql_files if 'fkindexes' not in f and 'schema' not in f]
    sql_files.sort()

    if limit:
        sql_files = sql_files[:limit]

    queries = []
    for sql_file in sql_files:
        query_name = os.path.basename(sql_file).replace('.sql', '')
        with open(sql_file, 'r') as f:
            sql = f.read().replace(';', '')
            queries.append({"name": query_name.upper(), "sql": sql})

    print(f"[*] Loaded {len(queries)} queries directly from job_d/")
    return queries


# --- Main Training & Simulation Loop ---

def main():
    parser = argparse.ArgumentParser(description="Bao Learned Optimizer")
    parser.add_argument("--limit", type=int, default=None, help="Limit total queries to process")
    parser.add_argument("--metrics-dir", type=str, default="metrics", help="Directory for metrics CSVs")
    args = parser.parse_args()

    queries = load_job_queries(args.limit)

    # Initialize Core Modules
    in_channels = len(PG_OPERATORS) + 4
    model = TreeCNN(in_channels=in_channels, out_channels=128)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    bandit = ThompsonSamplingBandit(model=model, num_mc_samples=5)
    replay_buffer = ExperienceReplayBuffer(capacity=2000)
    metrics = TrainingMetrics()

    print(f"[*] Starting Native Bao Continuous Training Loop...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[!] Critical Error: Could not connect to PostgreSQL. {e}")
        return

    with conn.cursor() as cur:
        total_queries_processed = 0

        # 1. Continuous Query Processing Loop (No fixed epochs)
        for q_id, q in enumerate(queries):
            total_queries_processed += 1
            training_cycle = ((total_queries_processed - 1) // 100) + 1

            # Routing Phase (Thompson Sampling)
            arm_plans = []
            for hint_idx, hint_str in BAO_HINT_SETS.items():
                explain_query = f"/*+ {hint_str} */ EXPLAIN (FORMAT JSON) {q['sql']}"
                try:
                    cur.execute(explain_query)
                    plan_json = cur.fetchone()[0][0]
                    plan_node = parse_plan_json(plan_json)
                    arm_plans.append(plan_node)
                except Exception as e:
                    conn.rollback()
                    arm_plans.append(None)

            best_arm_idx, predicted_log_time = bandit.select_arm(arm_plans)
            best_hint_str = BAO_HINT_SETS[best_arm_idx]
            optimal_plan_node = arm_plans[best_arm_idx]

            if optimal_plan_node is None:
                print(f"[!] Warning: Query {q['name']} produced no valid plans. Skipping.")
                continue

            cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)};")

            # Evaluate Native PostgreSQL (Baseline)
            cur.execute("DISCARD PLANS;")
            start_time = time.time()
            try:
                cur.execute(q['sql'])
                postgres_time_ms = (time.time() - start_time) * 1000
            except Exception:
                conn.rollback()
                postgres_time_ms = STATEMENT_TIMEOUT_MS
                cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)};")

            # Evaluate Bao Selected Hint
            bao_sql = f"/*+ {best_hint_str} */ {q['sql']}"
            cur.execute("DISCARD PLANS;")
            start_time = time.time()
            timed_out = False
            try:
                cur.execute(bao_sql)
                actual_time = (time.time() - start_time) * 1000
            except Exception:
                conn.rollback()
                cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)};")
                actual_time = STATEMENT_TIMEOUT_MS
                timed_out = True

            # Add to sliding window experience buffer
            replay_buffer.add(q_id, best_arm_idx, optimal_plan_node, actual_time)

            metrics.add_query_result(
                cycle=training_cycle,
                query_name=q["name"],
                query_id=q_id,
                selected_hint=best_arm_idx,
                predicted_log_time=predicted_log_time,
                actual_time_ms=actual_time,
                postgres_time_ms=postgres_time_ms,
                timed_out=timed_out,
                valid_plan_count=sum(1 for p in arm_plans if p is not None),
            )

            print(
                f"[{total_queries_processed}] Evaluated {q['name']:<4} | Hint: {best_arm_idx:2d} | Actual: {actual_time:7.2f} ms")

            # 2. Retrain the model every 100 queries
            if total_queries_processed % 100 == 0:
                print(f"\n[*] Initiating Retraining Sequence at query {total_queries_processed}...")
                model.train()

                # Sample |E| items with replacement (Bootstrap)
                batch_samples = replay_buffer.sample_with_replacement()
                dataset = TreeDataset(batch_samples)
                dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=tree_collate_fn)

                best_loss = float('inf')
                epochs_without_improvement = 0
                final_loss = 0.0

                # Train up to 100 epochs or until convergence
                for train_epoch in range(1, 101):
                    batch_loss_sum = 0
                    for plan_nodes, targets in dataloader:
                        optimizer.zero_grad()
                        preds = [model(node) for node in plan_nodes]
                        preds_tensor = torch.stack(preds)
                        loss = loss_fn(preds_tensor, targets)
                        loss.backward()
                        optimizer.step()
                        batch_loss_sum += loss.item()

                    avg_training_loss = batch_loss_sum / len(dataloader)
                    final_loss = avg_training_loss

                    # Convergence check: decrease < 1% over 10 epochs
                    if avg_training_loss < best_loss * 0.99:
                        best_loss = avg_training_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= 10:
                        print(f"    -> Convergence reached at epoch {train_epoch}. Loss: {avg_training_loss:.4f}")
                        break
                else:
                    print(f"    -> Max training epochs (100) reached. Final Loss: {final_loss:.4f}")

                metrics.add_training_loss(training_cycle, final_loss)
                metrics.summarize_cycle(training_cycle)
                print("\n")

    conn.close()
    print("\n[*] Training Sequence Complete.")
    metrics.print_final_summary()
    metrics.save_csvs(args.metrics_dir)

    os.makedirs("models", exist_ok=True)
    save_path = "models/bao_imdb.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[*] Native DB Training Weights secured in {save_path}")


if __name__ == "__main__":
    main()