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


class ExperienceReplayBuffer:
    def __init__(self, capacity=2000):
        self.capacity = capacity
        self.buffer = []

    def add(self, query_id, hint_set_idx, plan_node, actual_time_ms):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)

        self.buffer.append({
            'query_id': query_id,
            'hint_set_idx': hint_set_idx,
            'plan_node': plan_node,
            'actual_log_time': math.log(max(actual_time_ms, 1.0))
        })

    def sample_with_replacement(self):
        n_samples = len(self.buffer)
        return [random.choice(self.buffer) for _ in range(n_samples)]


class TrainingMetrics:
    def __init__(self):
        self.query_records = []
        self.epoch_records = []
        self.training_losses = []

    @staticmethod
    def _percentile(values, percentile):
        if not values: return float("nan")
        sorted_values = sorted(values)
        idx = int(math.ceil((percentile / 100.0) * len(sorted_values))) - 1
        return sorted_values[max(0, min(idx, len(sorted_values) - 1))]

    @staticmethod
    def _q_error(predicted_ms, actual_ms):
        predicted_ms = max(float(predicted_ms), 1.0)
        actual_ms = max(float(actual_ms), 1.0)
        return max(predicted_ms / actual_ms, actual_ms / predicted_ms)

    def add_query_result(
            self, cycle, query_name, query_id, selected_hint, predicted_log_time,
            actual_time_ms, postgres_time_ms, optimization_time_ms, timed_out, valid_plan_count
    ):
        predicted_time_ms = math.exp(float(predicted_log_time))
        q_error = self._q_error(predicted_time_ms, actual_time_ms)

        self.query_records.append({
            "epoch": cycle,
            "query_name": query_name,
            "query_id": query_id,
            "selected_hint": selected_hint,
            "predicted_log_time": float(predicted_log_time),
            "predicted_time_ms": predicted_time_ms,
            "actual_time_ms": actual_time_ms,
            "postgres_time_ms": postgres_time_ms,
            "optimization_time_ms": optimization_time_ms,
            "q_error": q_error,
            "timed_out": int(timed_out),
            "valid_plan_count": valid_plan_count,
        })

    def add_training_loss(self, cycle, loss):
        self.training_losses.append({"epoch": cycle, "training_loss": loss})

    def summarize_cycle(self, cycle):
        # Implementation remains the same
        pass

    def save_csvs(self, output_dir, file_prefix=""):
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{file_prefix}query_metrics.csv" if file_prefix else "query_metrics.csv"
        query_metrics_path = os.path.join(output_dir, filename)

        with open(query_metrics_path, "w", newline="") as f:
            fieldnames = [
                "epoch", "query_name", "query_id", "selected_hint", "predicted_log_time",
                "predicted_time_ms", "actual_time_ms", "postgres_time_ms", "optimization_time_ms",
                "q_error", "timed_out", "valid_plan_count",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.query_records)


class TreeDataset(Dataset):
    def __init__(self, buffer_samples): self.samples = buffer_samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]


def tree_collate_fn(batch):
    plan_nodes = [item['plan_node'] for item in batch]
    targets = torch.tensor([[item['actual_log_time']] for item in batch], dtype=torch.float32)
    return plan_nodes, targets


def load_job_queries(limit):
    sql_files = glob.glob('job_d/*.sql')
    sql_files = [f for f in sql_files if 'fkindexes' not in f and 'schema' not in f]
    sql_files.sort()
    if limit: sql_files = sql_files[:limit]

    queries = []
    for sql_file in sql_files:
        query_name = os.path.basename(sql_file).replace('.sql', '')
        with open(sql_file, 'r') as f:
            queries.append({"name": query_name.upper(), "sql": f.read().replace(';', '')})
    return queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics-dir", type=str, default="metrics")
    parser.add_argument("--port", type=int, default=5432)  # New argument
    parser.add_argument("--num-arms", type=int, default=48, help="Limit number of hint sets (1=Native PG)")
    args = parser.parse_args()

    # Apply the dynamic port
    DB_CONFIG["port"] = args.port

    queries = load_job_queries(args.limit)

    in_channels = len(PG_OPERATORS) + 4
    model = TreeCNN(in_channels=in_channels, out_channels=128)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    bandit = ThompsonSamplingBandit(model=model, num_mc_samples=5)
    replay_buffer = ExperienceReplayBuffer(capacity=2000)
    metrics = TrainingMetrics()

    # Slice the available arms based on CLI arg (to test Fig 12)
    available_arms = dict(list(BAO_HINT_SETS.items())[:args.num_arms])
    print(f"[*] Starting Native Bao Continuous Training Loop (Arms: {len(available_arms)})...")

    conn = psycopg2.connect(**DB_CONFIG)

    with conn.cursor() as cur:
        total_queries_processed = 0

        for q_id, q in enumerate(queries):
            total_queries_processed += 1
            training_cycle = ((total_queries_processed - 1) // 100) + 1

            # --- ROUTING/OPTIMIZATION PHASE ---
            opt_start_time = time.time()
            arm_plans = []

            for hint_idx, hint_str in available_arms.items():
                try:
                    cur.execute(f"/*+ {hint_str} */ EXPLAIN (FORMAT JSON) {q['sql']}")
                    plan_json = cur.fetchone()[0][0]
                    arm_plans.append(parse_plan_json(plan_json))
                except Exception:
                    conn.rollback()
                    arm_plans.append(None)

            best_arm_idx, predicted_log_time = bandit.select_arm(arm_plans)
            optimization_time_ms = (time.time() - opt_start_time) * 1000

            # Fallback for failed plans
            if arm_plans[best_arm_idx] is None: continue

            # --- EXECUTION PHASE ---
            best_hint_str = available_arms[best_arm_idx]
            cur.execute(f"SET statement_timeout = {int(STATEMENT_TIMEOUT_MS)};")

            # Native PG
            cur.execute("DISCARD PLANS;")
            pg_start = time.time()
            try:
                cur.execute(q['sql'])
                postgres_time_ms = (time.time() - pg_start) * 1000
            except Exception:
                conn.rollback()
                postgres_time_ms = STATEMENT_TIMEOUT_MS

            # Bao Execution
            cur.execute("DISCARD PLANS;")
            bao_start = time.time()
            timed_out = False
            try:
                cur.execute(f"/*+ {best_hint_str} */ {q['sql']}")
                actual_time = (time.time() - bao_start) * 1000
            except Exception:
                conn.rollback()
                actual_time = STATEMENT_TIMEOUT_MS
                timed_out = True

            # Tracking
            replay_buffer.add(q_id, best_arm_idx, arm_plans[best_arm_idx], actual_time)
            metrics.add_query_result(
                cycle=training_cycle, query_name=q["name"], query_id=q_id, selected_hint=best_arm_idx,
                predicted_log_time=predicted_log_time, actual_time_ms=actual_time,
                postgres_time_ms=postgres_time_ms, optimization_time_ms=optimization_time_ms,
                timed_out=timed_out, valid_plan_count=sum(1 for p in arm_plans if p is not None)
            )

            # ... Retraining logic (unchanged) ...
            # Retrain every 100 queries
            if total_queries_processed % 100 == 0:
                model.train()
                batch_samples = replay_buffer.sample_with_replacement()
                dataset = TreeDataset(batch_samples)
                dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=tree_collate_fn)

                best_loss = float('inf')
                epochs_without_improvement = 0
                for train_epoch in range(1, 101):
                    batch_loss_sum = 0
                    for plan_nodes, targets in dataloader:
                        optimizer.zero_grad()
                        preds = [model(node) for node in plan_nodes]
                        loss = loss_fn(torch.stack(preds), targets)
                        loss.backward()
                        optimizer.step()
                        batch_loss_sum += loss.item()

                    avg_training_loss = batch_loss_sum / len(dataloader)
                    print(f"Epoch {train_epoch}: Training Loss = {avg_training_loss:.4f}")
                    if avg_training_loss < best_loss * 0.99:
                        best_loss = avg_training_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= 10: break

    conn.close()
    os.makedirs(args.metrics_dir, exist_ok=True)
    prefix = f"{args.num_arms}_arms_" if args.num_arms != 48 else ""
    metrics.save_csvs(args.metrics_dir, file_prefix=prefix)


if __name__ == "__main__":
    main()