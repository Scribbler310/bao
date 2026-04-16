# Bao: Bandit Learned Optimizer for PostgreSQL 

A faithful, functional re-creation of the **Bao Query Optimizer** ([SIGMOD 2021](https://rmarcus.info/bao.html)), implemented in complete end-to-end Python using **PyTorch**, **PostgreSQL**, and **pg_hint_plan**.

## Motivation
Bao functions as a contextual multi-armed bandit. Instead of predicting database costs to form execution paths manually, it selects a set of robust optimization hints using Thompson Sampling, captures the actual execution topological structure through PostgreSQL `EXPLAIN FORMAT JSON`, and feeds the execution timings to a Tree Convolutional Neural Network (TreeCNN) for real-time workload learning!

---

## 🛠️ Installation & Setup (From Scratch)

This repository is designed specifically to optimize the massive Join Order Benchmark (JOB). To reconstruct the environment from scratch natively on your local machine, follow these instructions strictly:

### 1. Database Dependencies & Docker
Ensure that you have [Docker & Docker Compose](https://www.docker.com/) installed on your machine.
A specialized `postgres:15` image with `pg_hint_plan` compiled statically is configured in the `.yml`. Start the PostgreSQL testing network:
```bash
docker-compose up -d
```

### 2. Prepare the Workload Datasets
The training pipeline requires the 4GB IMDB dataset and the 113 analytical Join Order Benchmark queries.

**A. Download IMDB Database (Data)**
Bao requires real data to optimize. Download the official IMDB CSV dataset and place all extracted `.csv` files and `schematext.sql` securely into an `imdb` folder located *one level above* this project root. Your file structure should look like this:
```
/imdb/
  ├── title.csv
  ├── company_name.csv
  ├── schematext.sql
  └── (18 other csvs...)
/bao_project/
  ├── docker-compose.yml
  └── main.py
```

**B. Download JOB Queries (Workload)**
Next, download the Join Order Benchmark `.sql` queries. Execute this natively inside `/bao_project`:
```bash
git clone https://github.com/gregrahn/join-order-benchmark.git job_queries
```

### 3. Setup Python Pipeline
With Python 3.10+, spawn a dedicated virtual environment and install the neural networking and database connectivity matrices natively:
```bash
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Mac/Linux)
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 🚀 Execution & Training 

### Model Training (`main.py`)
To train your TreeCNN against PostgreSQL natively:
```bash
# This forces Bao to execute real analytical workloads, dynamically timing Postgres responses
python main.py --limit 3 --epochs 2
```
*Note: We highly recommend using `--limit` or heavily caching. Exploring sub-optimal execution paths statically on the entire 113-query JOB dataset natively without constraints could take tens of hours in early epochs.*

### Model Evaluation (`test.py`)
To test what your trained neural network learned:
```bash
python test.py --limit 10
```
This script sequentially loads your `models/bao_imdb.pt`, natively pulls topological layouts for all hints via Postgres, selects the optimal `Bao Hint`, benchmarks against the baseline Postgres, and exports the latency analytics securely to a spreadsheet viewable `benchmark_results.csv` output natively!
