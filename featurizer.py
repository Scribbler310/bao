import json
import math
import numpy as np

# A representative set of PostgreSQL physical operators
PG_OPERATORS = [
    'Aggregate', 'Gather', 'Gather Merge', 'Hash Join', 'Merge Join', 'Nested Loop',
    'Seq Scan', 'Index Scan', 'Index Only Scan', 'Bitmap Heap Scan', 'Bitmap Index Scan',
    'Parallel Seq Scan', 'Parallel Index Scan', 'Parallel Index Only Scan', 'Parallel Bitmap Heap Scan',
    'Sort', 'Hash', 'Materialize', 'Limit', 'Subquery Scan', 'CTE Scan', 'Result',
    'Append', 'Merge Append', 'Recursive Union', 'WindowAgg', 'Group', 'LockRows',
    'SetOp', 'Unique', 'Foreign Scan', 'Custom Scan', 'Values Scan', 'Function Scan',
    'Table Function Scan', 'WorkTable Scan', 'Named Tuplestore Scan', 'CTE Scan',
    'Project Set', 'Incremental Sort', 'Memoize', 'Tid Scan', 'Tid Range Scan',
    'Sample Scan'
]

# Map operator to one-hot index
OP_TO_IDX = {op: i for i, op in enumerate(PG_OPERATORS)}

class PlanNode:
    def __init__(self, node_type, startup_cost, total_cost, plan_rows, plan_width, children=None):
        self.node_type = node_type
        self.startup_cost = startup_cost
        self.total_cost = total_cost
        self.plan_rows = plan_rows
        self.plan_width = plan_width
        self.children = children if children else []

    def get_feature_vector(self):
        """
        Create the node-level vector representation.
        Features = One-hot encoded node type + Log-scaled metrics
        """
        # 1. One-hot encode node type
        one_hot = np.zeros(len(PG_OPERATORS))
        idx = OP_TO_IDX.get(self.node_type)
        if idx is not None:
            one_hot[idx] = 1.0
            
        # 2. Log-normalize metrics as requested: log(x + 1)
        metrics = np.array([
            math.log(self.startup_cost + 1),
            math.log(self.total_cost + 1),
            math.log(self.plan_rows + 1),
            math.log(self.plan_width + 1)
        ])
        
        # Concatenate features
        return np.concatenate([one_hot, metrics])

def parse_plan_json(plan_dict):
    """
    Recursively parse the PostgreSQL EXPLAIN FORMAT JSON dictionary 
    into a tree of PlanNode objects.
    """
    # The JSON usually wraps the actual plan in a "Plan" key
    if 'Plan' in plan_dict:
        plan_dict = plan_dict['Plan']
        
    node_type = plan_dict.get('Node Type', 'Unknown')
    startup_cost = plan_dict.get('Startup Cost', 0.0)
    total_cost = plan_dict.get('Total Cost', 0.0)
    plan_rows = plan_dict.get('Plan Rows', 0.0)
    plan_width = plan_dict.get('Plan Width', 0.0)
    
    # Recursively parse children (Plans)
    children = []
    if 'Plans' in plan_dict:
        for child_dict in plan_dict['Plans']:
            children.append(parse_plan_json(child_dict))
            
    return PlanNode(node_type, startup_cost, total_cost, plan_rows, plan_width, children)
