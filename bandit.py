import torch
import numpy as np

import itertools

# List of PostgreSQL planner method flags to manipulate
PLANNER_FLAGS = [
    "enable_nestloop", "enable_mergejoin", "enable_hashjoin",
    "enable_indexscan", "enable_indexonlyscan", "enable_bitmapscan", "enable_seqscan"
]

def generate_hint_sets():
    """
    Generate 48 systematically varied hint sets (arms) by disabling subsets
    of join types and scan types, mimicking the Bao paper methodology.
    """
    hint_sets = {0: "Set(enable_hashjoin on) Set(enable_mergejoin on) Set(enable_nestloop on)"}
    
    # Generate all combinations of flags to disable
    all_combos = []
    # 1 flag off, 2 flags off, 3 flags off...
    for r in range(1, 4):
        for combo in itertools.combinations(PLANNER_FLAGS, r):
            # Constraint: Don't disable all join types or all scan types
            off_set = set(combo)
            joins = {"enable_nestloop", "enable_mergejoin", "enable_hashjoin"}
            scans = {"enable_indexscan", "enable_indexonlyscan", "enable_bitmapscan", "enable_seqscan"}
            
            if off_set.issuperset(joins) or off_set.issuperset(scans):
                continue
            all_combos.append(combo)
            
    # Select the first 47 combinations to get 48 total arms (including index 0)
    for i, combo in enumerate(all_combos[:47]):
        hint_str = " ".join([f"Set({flag} off)" for flag in combo])
        hint_sets[i+1] = hint_str
        
    return hint_sets

BAO_HINT_SETS = generate_hint_sets()

class ThompsonSamplingBandit:
    def __init__(self, model, num_mc_samples=10):
        """
        model: The PyTorch TreeCNN model
        num_mc_samples: Number of forward passes to perform for Monte Carlo Dropout
                        which estimates the posterior distribution.
        """
        self.model = model
        self.num_mc_samples = num_mc_samples

    def select_arm(self, arm_plans):
        """
        Given a list of plan objects (one for each arm), select the best
        arm using Thompson Sampling via Monte Carlo Dropout.
        
        arm_plans: list of PlanNode (these are the EXPLAIN plan trees for each hint set)
        returns: (best_arm_index, predicted_latency)
        """
        # Enable dropout to sample from the model's predictive posterior
        self.model.train()
        
        arm_samples = []
        for plan_node in arm_plans:
            # We must skip if a plan could not be generated (e.g. PG forced a bad plan and gave error, though EXPLAIN usually succeeds)
            if plan_node is None:
                # Assign incredibly high predicted cost to invalid plans
                arm_samples.append([float('inf')])
                continue
                
            samples = []
            for _ in range(self.num_mc_samples):
                with torch.no_grad():
                    pred_log_time = self.model(plan_node)
                    samples.append(pred_log_time.item())
            arm_samples.append(samples)
            
        # Draw one sample from each arm's posterior (in MC dropout, we just take one randomly from our drawn samples)
        # or we calculate mean/var and sample from a normal distribution.
        # Following simple Thompson Sampling logic: draw a sample for each arm and pick the minimum.
        drawn_latencies = []
        for samples in arm_samples:
            if samples[0] == float('inf'):
                drawn_latencies.append(float('inf'))
            else:
                # Randomly pick one of the MC samples
                drawn_sample = np.random.choice(samples)
                drawn_latencies.append(drawn_sample)
                
        best_arm_idx = np.argmin(drawn_latencies)
        # Return the chosen arm and its expected latency (mean of MC samples)
        expected_latency = np.mean(arm_samples[best_arm_idx]) if drawn_latencies[best_arm_idx] != float('inf') else float('inf')
        
        return best_arm_idx, expected_latency
