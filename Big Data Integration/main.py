import os
import numpy as np
import pandas as pd
import csv
from argparse import ArgumentParser
from prepare_sources import prepare_source
from blocking import train_fastText, vectorize_instances,  build_LSH_tables, define_pw_matchings_to_perform

def get_args():
    parser = ArgumentParser(description='Hyperparameters', add_help=True)
    parser.add_argument('-dataset', type=str, help='Dataset to Elaborate')
    parser.add_argument('-ls_dim', type=int, default=256, help='fastText Embedding Dimension')
    parser.add_argument('-p1', type=float, default=0.6, help='Blocking P1 parameter')
    parser.add_argument('-p2', type=float, default=0.15, help='Blocking P2 parameter')

    return parser.parse_args()

### #### ###
### MAIN ###
### #### ###
args = get_args()
datasets = ['Abt_Buy', 'Beers', 'DBLP_ACM']

CWD = os.getcwd()
DATASET, LS_DIM, P1, P2 = args.dataset, args.ls_dim, args.p1, args.p2

assert DATASET in datasets, "Invalid Dataset: choose among the Datasets in './datasets/'"
assert P1 >= 0 and P1 <= 1, "Invalid Blocking Parameter: P1 is a probability and must be in [0, 1]"
assert P2 >= 0 and P2 <= 1, "Invalid Blocking Parameter: P2 is a probability and must be in [0, 1]"
assert P1 > P2, "Invalid Blocking Parameters: P1 must be greater than P2"

### PHASE 1 -> DATA PREPARATION ###
print("*** PHASE 1 -> SOURCE PREPARATION ***")
if not os.path.exists(f'./datasets/{DATASET}/instances_refined.csv'):
    print(f"Preparing Source: {DATASET}")
    prepare_source(DATASET)
else:
    print(f"Source Already Prepared: {DATASET}")

### PHASE 2 -> BLOCKING ###
print("\n*** PHASE 2 -> BLOCKING ***")
instances = pd.read_csv(f'./datasets/{DATASET}/instances_refined.csv', index_col=0, header=0, dtype=str)

for col in instances.columns:
    instances[col] = instances[col].astype(str)

word_weights = pd.read_csv(f'./datasets/{DATASET}/word_weights.csv', index_col=0, header=0)

ft_model = train_fastText(instances, LS_DIM, DATASET)
instance_vectors = vectorize_instances(instances, ft_model, word_weights)

rho = np.log(1/P1) / np.log(1/P2)
L = round(len(instances)**rho)
K = round(np.log(len(instances)) / np.log(1/P2))

if L == 0:
    L = 1
if K == 0:
    K = 1

tables = build_LSH_tables(instance_vectors, L, K, LS_DIM)
define_pw_matchings_to_perform(tables, DATASET)

gt = pd.read_csv(f'./datasets/{DATASET}/ground_truth.csv', header=0)
tn, fn, fp, tp = 0, 0, 0, 0

matchings_to_perform = 0
with open(f"./datasets/{DATASET}/pw_matchings_to_perform.csv", "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        matchings_to_perform += 1
        i1, i2 = row
        for idx in gt.index:
            if (i1, i2) == (gt.at[idx, 'ltable_id'], gt.at[idx, 'rtable_id']) or (i2, i1) == (gt.at[idx, 'ltable_id'], gt.at[idx, 'rtable_id']):
                if gt.at[idx, 'label'] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                if gt.at[idx, 'label'] == 1:
                    fn += 1
                else:
                    tn += 1


print(f"Total Comparisons Baseline: {int(len(instances) * (len(instances) - 1) / 2)}")
print(f"Total Comparisons to Perform: {matchings_to_perform}")
print(f"Percentual Reduction: {100 - (100*matchings_to_perform / (len(instances) * (len(instances) - 1) / 2))}%")

recall = tp / (tp + fn)
precision = tp / (tp + fp)
f_measure = 2 * (precision * recall) / (precision + recall)

print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F-Measure: {f_measure}")

### PHASE 3 -> PAIRWISE MATCHING ###


### PHASE 4 -> CLUSTERING ###