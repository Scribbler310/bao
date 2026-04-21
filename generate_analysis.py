import os
import csv
import json
import time
import torch
import psycopg2
import ollama
import math
from model import TreeCNN
from featurizer import parse_plan_json, PG_OPERATORS
from bandit import BAO_HINT_SETS

# DB Configuration from test.py
DB_CONFIG = {
    "dbname": "bao",
    "user": "bao",
    "password": "bao",
    "host": "localhost",
    "port": 5432
}

def get_query_sql(query_name):
    """Load the SQL for a given JOB query name."""
    # Handle cases where Query_Name might have prefixes or suffixes
    sql_path = os.path.join("job_queries", f"{query_name.lower()}.sql")
    if not os.path.exists(sql_path):
        # Try upper case just in case
        sql_path = os.path.join("job_queries", f"{query_name.upper()}.sql")
    
    if os.path.exists(sql_path):
        with open(sql_path, 'r') as f:
            return f.read().replace(';', '')
    return None

def get_plan_details(conn, sql, hint):
    """Retrieve the explain plan and extract key metrics."""
    with conn.cursor() as cur:
        explain_query = f"/*+ {hint} */ EXPLAIN (FORMAT JSON) {sql}"
        try:
            cur.execute(explain_query)
            plan_json = cur.fetchone()[0][0]
            
            # The top-level plan info
            top_level = plan_json.get('Plan', {})
            node_type = top_level.get('Node Type', 'Unknown')
            plan_rows = top_level.get('Plan Rows', 0)
            
            return plan_json, node_type, plan_rows
        except Exception as e:
            print(f"Error running EXPLAIN: {e}")
            conn.rollback()
            return None, None, None

def generate_analysis():
    print("[*] Starting Bao result analysis pipeline...")
    
    # 1. Load Model
    model_path = os.path.join("models", "bao_imdb.pt")
    in_channels = len(PG_OPERATORS) + 4
    model = TreeCNN(in_channels=in_channels, out_channels=128)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"[*] Loaded TreeCNN weights from {model_path}")
    else:
        print("[!] Warning: Model weights not found. Predictions will be random.")

    # 2. Connect to DB
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("[*] Connected to PostgreSQL instance.")
    except Exception as e:
        print(f"[!] Failed to connect to DB: {e}")
        return

    # 3. Read Benchmark Results
    results_file = "benchmark_results.csv"
    if not os.path.exists(results_file):
        print(f"[!] {results_file} not found.")
        return

    rows = []
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if 'AI_Summary' not in fieldnames:
        fieldnames.append('AI_Summary')
    if 'Predicted_Time_ms' not in fieldnames:
        fieldnames.append('Predicted_Time_ms')

    print(f"[*] Processing {len(rows)} queries...")

    # Open bao_insights.md for appending
    with open("bao_insights.md", "w") as md_file:
        md_file.write("# Bao Query Optimizer: Strategic Insights\n\n")
        md_file.write("| Query | Hint | Postgres (ms) | Actual (ms) | Summary |\n")
        md_file.write("|---|---|---|---|---|\n")

        for idx, row in enumerate(rows):
            query_name = row['Query_Name']
            hint = row['Bao_Hint']
            actual_time = row['Bao_Time_ms']
            postgres_time = row['Postgres_Time_ms']
            
            print(f"    -> Analyzing {query_name} ({idx+1}/{len(rows)})...")
            
            sql = get_query_sql(query_name)
            if not sql:
                print(f"       [!] SQL not found for {query_name}")
                continue
                
            plan_json, node_type, plan_rows = get_plan_details(conn, sql, hint)
            if not plan_json:
                continue

            # 4. Re-predict Latency
            plan_node = parse_plan_json(plan_json)
            with torch.no_grad():
                pred_log_time = model(plan_node)
                predicted_time = math.exp(pred_log_time.item())
            
            row['Predicted_Time_ms'] = f"{predicted_time:.2f}"

            # 5. Call Gemma 4 via Ollama
            # Skip if we already have a summary, unless user wants to overwrite
            if row.get('AI_Summary') and len(row['AI_Summary']) > 10:
                print(f"       [ ] Skipping {query_name} (already summarized)")
                continue

            # Cast to float for calculation in the prompt
            f_actual = float(actual_time)
            f_pg = float(postgres_time or 1.0)
            speedup = f_pg / f_actual if f_actual > 0 else 0

            prompt = (
                f"Analyze this Bao result comparison for Query {query_name}:\n"
                f"Bao Selected Hint: {hint}\n"
                f"Bao Top Node: {node_type}\n"
                f"Actual Bao Latency: {f_actual:.2f}ms\n"
                f"Native PostgreSQL Latency: {f_pg:.2f}ms\n\n"
                f"Task: Focus on the performance difference between Native Postgres and the Bao-optimized path. "
                f"Explain why this specific hint configuration ({hint}) produced the observed {speedup:.2f}x speedup (or slowdown). "
                f"Provide a STRICT ONE-SENTENCE strategic summary (max 35 words), avoiding bolding or conversational filler."
            )

            try:
                response = ollama.generate(model='gemma4:e4b', prompt=prompt)
                summary = response['response'].strip().replace('\n', ' ')
            except Exception as e:
                print(f"       [!] Ollama Error: {e}")
                summary = "Error generating summary."

            row['AI_Summary'] = summary
            
            # Update Markdown
            md_file.write(f"| {query_name} | `{hint[:30]}...` | {postgres_time} | {actual_time} | {summary} |\n")
            md_file.flush()

    # 6. Write back to CSV
    with open(results_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    conn.close()
    print("\n[*] Analysis complete! Updated benchmark_results.csv and generated bao_insights.md")

if __name__ == "__main__":
    generate_analysis()
