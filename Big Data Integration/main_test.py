import os
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
from tqdm import tqdm
from prepare_sources import prepare_source
from blocking import train_fastText, vectorize_instances,  build_LSH_tables, define_pw_matchings_to_perform
from clustering import instantiate_llm, build_pw_prompt, query_llm

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
if not os.path.exists(f'./datasets/{DATASET}/instances_refined.csv'):
    print(f"Preparing Source: {DATASET}")
    prepare_source(DATASET)
else:
    print(f"Source Already Prepared: {DATASET}")

### PHASE 2 -> BLOCKING ###
instances = pd.read_csv(f'./datasets/{DATASET}/instances_refined.csv', index_col=0, header=0, dtype=str)
for col in instances.columns:
    instances[col] = instances[col].astype(str)

word_weights = pd.read_csv(f'./datasets/{DATASET}/word_weights.csv', index_col=0, header=0)

ft_model = train_fastText(instances, LS_DIM, DATASET)
instance_vectors = vectorize_instances(instances, ft_model, word_weights, DATASET)

rho = np.log(1/P1) / np.log(1/P2)
L = round(len(instances)**rho)
K = round(np.log(len(instances)) / np.log(1/P2))

if L == 0:
    L = 1
if K == 0:
    K = 1

tables = build_LSH_tables(instance_vectors, L, K, LS_DIM)
define_pw_matchings_to_perform(tables, DATASET)

### PHASE 3 -> QUERYING LLM ###
llm = instantiate_llm()
rl_prompt = build_pw_prompt()

tn, fn, fp, tp = 0, 0, 0, 0
gt = pd.read_csv(f"./datasets/{DATASET}/ground_truth.csv", header=0)
valid_pairs = pd.read_csv(f"./datasets/{DATASET}/pw_matchings_to_perform.csv", header=None)
valid_pairs_set = set(zip(valid_pairs[0], valid_pairs[1]))

for idx in tqdm(gt.index):
    i1, i2, label = gt.loc[idx]
    same_block = (((i1, i2) in valid_pairs_set) or ((i2, i1) in valid_pairs_set))
    if label == 0:
        if same_block == False:
            tn += 1
        else:
            resp = query_llm(llm, rl_prompt, DATASET, instances, i1, i2)
            if 'MATCH' in resp:
                fp += 1
            else:
                tn += 1
    else:
        if same_block == False:
            fn += 1
        else:
            resp = query_llm(llm, rl_prompt, DATASET, instances, i1, i2)
            if 'MATCH' in resp:
                tp += 1
            else:
                fn += 1

recall = tp / (tp + fn)
precision = tp / (tp + fp)
f_measure = (2 * recall * precision) / (recall + precision)

print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F-Measure: {f_measure}")

torch.cuda.empty_cache()