import os
import time
import argparse
import csv
import psycopg2

from bandit import BAO_HINT_SETS
from test import load_job_queries, check_and_seed_data, DB_CONFIG


def precompute_optimals(limit, max_timeout_ms=20000):
    print("[*] Connecting to PostgreSQL...")
    try:
        conn = psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"[!] Could not connect: {e}")
        return

    check_and_seed_data(conn)

    # Load ONLY the test holdout set
    test_queries = load_job_queries(limit=limit, mode="test")
    print(f"[*] Loaded {len(test_queries)} holdout queries for optimal baseline search.")

    os.makedirs("metrics", exist_ok=True)
    out_file = os.path.join("metrics", "optimal_baselines.csv")

    with conn.cursor() as cur:
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["query_name", "postgres_time_ms", "optimal_time_ms", "optimal_hint"])

            for q in test_queries:
                print(f"\n--- Searching Optimal for {q['name']} ---")

                # 1. Evaluate baseline PostgreSQL
                cur.execute(f"SET statement_timeout = {max_timeout_ms};")
                cur.execute("DISCARD PLANS;")

                start_time = time.time()
                try:
                    cur.execute(q['sql'])
                    pg_time = (time.time() - start_time) * 1000
                except psycopg2.errors.QueryCanceled:
                    pg_time = max_timeout_ms
                    conn.rollback()
                except Exception as e:
                    pg_time = max_timeout_ms
                    conn.rollback()

                print(f"    -> Native PG Time: {pg_time:.2f} ms")

                best_time = pg_time
                best_hint = "Native"

                # 2. Evaluate all 48 Bao hints
                for hint_idx, hint_str in BAO_HINT_SETS.items():
                    # DYNAMIC TIMEOUT: If a plan takes 1.2x longer than our current best, kill it.
                    # This saves massive amounts of time on terrible execution plans.
                    dynamic_timeout = int(min(max_timeout_ms, best_time * 1.2))
                    dynamic_timeout = max(dynamic_timeout, 100)  # enforce minimum 100ms timeout safety

                    try:
                        cur.execute(f"SET statement_timeout = {dynamic_timeout};")
                        cur.execute("DISCARD PLANS;")

                        start_time = time.time()
                        bao_sql = f"/*+ {hint_str} */ {q['sql']}"
                        cur.execute(bao_sql)

                        hint_time = (time.time() - start_time) * 1000

                        if hint_time < best_time:
                            best_time = hint_time
                            best_hint = hint_str
                            print(f"       [New Best] Hint {hint_idx} | Time: {best_time:.2f} ms")

                    except psycopg2.errors.QueryCanceled:
                        # The plan was slower than our dynamic timeout, skip it
                        conn.rollback()
                    except Exception as e:
                        # Invalid query plan, skip it
                        conn.rollback()

                print(f"    -> Optimal Found:  {best_time:.2f} ms")
                writer.writerow([q['name'], pg_time, best_time, best_hint])
                f.flush()  # Save incrementally in case the script crashes

    conn.close()
    print(f"\n[*] Finished! Saved optimal baselines to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute Optimal Bao Hints")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    precompute_optimals(args.limit)