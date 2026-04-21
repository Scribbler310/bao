import os
import time
import psycopg2
import torch
import glob
import argparse
import csv
from model import TreeCNN
from featurizer import parse_plan_json, PG_OPERATORS
from bandit import BAO_HINT_SETS

DB_CONFIG = {
    "dbname": "bao",
    "user": "bao",
    "password": "bao",
    "host": "localhost",
    "port": 5432
}

def check_and_seed_data(conn):
    """Seed the entire database with IMDB data if missing."""
    print("[*] Checking if database requires seeding...")
    with conn.cursor() as cur:
        # Check if title table has data
        cur.execute("SELECT count(*) FROM title;")
        title_count = cur.fetchone()[0]
        
        if title_count == 0:
            print("[*] Missing data. Seeding full schema and CSVs...")
            with open('../imdb/schematext.sql', 'r') as f:
                schema_sql = f.read()
            try:
                cur.execute(schema_sql)
                conn.commit()
            except Exception as e:
                print(f"[*] Schema potentially already exists. Continuing to COPY... ({e})")
                conn.rollback()

            print("[*] Seeding all CSV data via COPY into IMDB schema...")
            # We iterate over the local ../imdb directory to find csv names
            csv_files = glob.glob('../imdb/*.csv')
            for csv_path in csv_files:
                filename = os.path.basename(csv_path)
                table_name = filename.replace('.csv', '')
                print(f"    -> Loading {table_name}...")
                try:
                    # Note: we use /imdb/ natively inside the docker postgres process
                    cur.execute(f"COPY {table_name} FROM '/imdb/{filename}' WITH (FORMAT csv, DELIMITER ',', ESCAPE '\\');")
                    conn.commit()
                except Exception as e:
                    print(f"[!] Warning during COPY of {table_name}: {e}. Skipping table.")
                    conn.rollback()
            
            print("[*] All Data successfully seeded. Building Indexes...")
            
            # Optionally build indexes to prevent query hangs
            if os.path.exists('job_queries/fkindexes.sql'):
                try:
                    with open('job_queries/fkindexes.sql', 'r') as f:
                        index_sql = f.read()
                    cur.execute(index_sql)
                    conn.commit()
                    print("[*] Indexes built successfully.")
                except Exception as e:
                    print(f"[!] Warning during Index build: {e}")
                    conn.rollback()
        else:
            print("[*] Database is already seeded.")

def load_job_queries(limit, split_ratio=0.2, seed=42, mode="test"):
    """
    Load JOB queries and split them by template. 
    Mode 'test' returns the hold-out set, Mode 'train' returns the training set.
    """
    import re
    import random
    from collections import defaultdict
    
    sql_files = glob.glob('job_queries/*.sql')
    sql_files = [f for f in sql_files if 'fkindexes' not in f and 'schema' not in f]
    sql_files.sort()
    
    # Group queries by template (e.g. 1a.sql, 1b.sql -> template 1)
    template_groups = defaultdict(list)
    for f in sql_files:
        name = os.path.basename(f)
        template_id = re.match(r'(\d+)', name).group(1)
        template_groups[template_id].append(f)
        
    unique_templates = sorted(list(template_groups.keys()), key=int)
    random.seed(seed)
    random.shuffle(unique_templates)
    
    # The split ratio here is for the TRAINING set
    # If split_ratio=0.8, then 80% is train, 20% is test.
    # In test mode, we take the latter part.
    train_split_ratio = 1.0 - split_ratio if mode == "test" else split_ratio
    split_idx = int(len(unique_templates) * (1.0 - split_ratio))
    
    if mode == "test":
        selected_templates = unique_templates[split_idx:]
    else:
        selected_templates = unique_templates[:split_idx]
        
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

def evaluate_queries(limit, split_ratio, seed):
    print("[*] Connecting to PostgreSQL...")
    conn = psycopg2.connect(**DB_CONFIG)
    
    check_and_seed_data(conn)
    
    print("[*] Loading trained Bao Pytorch Model...")
    model_path = os.path.join("models", "bao_imdb.pt")
    
    in_channels = len(PG_OPERATORS) + 4
    model = TreeCNN(in_channels=in_channels, out_channels=128)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"[*] Loaded weights from {model_path}.")
    else:
        print("[!] No trained weights found. Model will predict randomly.")
        
    test_queries = load_job_queries(limit=limit, split_ratio=split_ratio, seed=seed, mode="test")
    print(f"[*] Loaded {len(test_queries)} JOB Benchmarks for evaluation.")
    
    with conn.cursor() as cur:
        # Open CSV file for writing results
        csv_file = open("benchmark_results.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Query_Name", "Postgres_Hint", "Bao_Hint", "Postgres_Time_ms", "Bao_Time_ms", "Speedup"])
        
        for q in test_queries:
            print(f"\n--- Benchmarking JOB Query: {q['name']} ---")
            
            best_hint = None
            best_predicted_latency = float('inf')
            
            # Step 1: Query PostgreSQL for EXPLAIN JSON for evaluating all hints
            for hint_idx, hint_str in BAO_HINT_SETS.items():
                explain_query = f"/*+ {hint_str} */ EXPLAIN (FORMAT JSON) {q['sql']}"
                try:
                    cur.execute(explain_query)
                    plan_json = cur.fetchone()[0][0]
                except Exception as e:
                    print(f"Failed EXPLAIN for hint {hint_idx}: {e}")
                    conn.rollback()
                    continue
                
                plan_node = parse_plan_json(plan_json)
                
                with torch.no_grad():
                    pred_latency_var = model(plan_node)
                    pred_latency = pred_latency_var.item()
                
                if pred_latency < best_predicted_latency:
                    best_predicted_latency = pred_latency
                    best_hint = hint_str
            
            if not best_hint:
                print("[!] Could not parse EXPLAIN plans for query. Skipping execution.")
                continue
                
            print(f"[*] Bao Selected Best Hint: {best_hint[:50]}... (Pred Latency: {best_predicted_latency:.4f})")
            
            baseline_sql = q['sql']
            bao_sql = f"/*+ {best_hint} */ {q['sql']}"
            
            # Set timeout to prevent extremely bad natively executed plans from hanging the test loop forever (15s limit)
            cur.execute("SET statement_timeout = 15000;") 
            
            # Execute Baseline
            cur.execute("DISCARD PLANS;") # clear cache
            start_time = time.time()
            try:
                cur.execute(baseline_sql)
                baseline_time = (time.time() - start_time) * 1000
                print(f"    -> Baseline PostgreSQL Latency: {baseline_time:.2f} ms")
            except Exception as e:
                print(f"    -> Baseline PostgreSQL Timed Out (>15s) or Failed.")
                baseline_time = 15000
                conn.rollback()
                cur.execute("SET statement_timeout = 15000;") 
            
            # Execute Bao Optimized
            cur.execute("DISCARD PLANS;")
            start_time = time.time()
            try:
                cur.execute(bao_sql)
                bao_time = (time.time() - start_time) * 1000
                print(f"    -> Bao Optimized Latency:       {bao_time:.2f} ms")
            except Exception as e:
                print(f"    -> Bao Optimized Timed Out (>15s) or Failed.")
                bao_time = 15000
                conn.rollback()

            speedup = baseline_time / max(bao_time, 0.001)
            print(f"    -> Speedup: {speedup:.2f}x")
            
            # Log to CSV
            csv_writer.writerow([
                q['name'], 
                "Native (None)", 
                best_hint, 
                f"{baseline_time:.2f}", 
                f"{bao_time:.2f}", 
                f"{speedup:.2f}"
            ])
            csv_file.flush()

    csv_file.close()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Bao on JOB")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of JOB queries tested")
    parser.add_argument("--split", type=float, default=0.2, help="Hold-out split ratio (templates, default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (must match main.py)")
    args = parser.parse_args()
    evaluate_queries(args.limit, args.split, args.seed)
