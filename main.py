import os
import argparse
import random
import time
import math
import glob
import torch
import psycopg2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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
    args = parser.parse_args()
    
    queries = load_job_queries(args.limit, split_ratio=args.split, seed=args.seed, mode="train")
    
    # Initialize Core Modules
    in_channels = len(PG_OPERATORS) + 4
    model = TreeCNN(in_channels=in_channels, out_channels=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    
    bandit = ThompsonSamplingBandit(model=model, num_mc_samples=5)
    replay_buffer = ExperienceReplayBuffer(capacity=5000)

    print(f"[*] Starting Native Bao Training Loop on {len(queries)} JOB Benchmark Queries...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[!] Critical Error: Could not connect to PostgreSQL. {e}")
        return

    with conn.cursor() as cur:
        for epoch in range(args.epochs):
            print(f"--- Epoch {epoch + 1}/{args.epochs} ---")
            
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
                cur.execute("SET statement_timeout = 20000;")
                bao_sql = f"/*+ {best_hint_str} */ {q['sql']}"
                
                cur.execute("DISCARD PLANS;") # Always flush caches for accurate learning
                start_time = time.time()
                try:
                    cur.execute(bao_sql)
                    actual_time = (time.time() - start_time) * 1000
                except psycopg2.errors.QueryCanceled:
                    print(f"    -> Query {q['name']} timed out using Hint {best_arm_idx}. Applying heavy cost penalty.")
                    conn.rollback()
                    cur.execute("SET statement_timeout = 20000;")
                    actual_time = 20000.0 # Time threshold penalty
                except Exception as e:
                    print(f"    -> Query {q['name']} failed execution using Hint {best_arm_idx}: {e}")
                    conn.rollback()
                    cur.execute("SET statement_timeout = 20000;")
                    actual_time = 20000.0 # Safety failure penalty
                
                # Store structural truth inside Replay Buffer
                replay_buffer.add(q_id, best_arm_idx, optimal_plan_node, actual_time)
                print(f"[*] Evaluated Query {q['name']:<4} | Hint Selected: {best_arm_idx} | Exec Time: {actual_time:7.2f} ms")
                
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
                    preds_tensor = torch.stack(preds) # [batch_size, 1]
                    
                    loss = loss_fn(preds_tensor, targets)
                    loss.backward()
                    optimizer.step()
                    
                    batch_loss_sum += loss.item()
                    
                print(f"[*] Training Loss: {batch_loss_sum / len(dataloader):.4f}")

    conn.close()
    
    print("\n[*] Training Sequence Complete.")
    
    os.makedirs("models", exist_ok=True)
    save_path = "models/bao_imdb.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[*] Native DB Training Weights secured in {save_path}")

if __name__ == "__main__":
    main()
