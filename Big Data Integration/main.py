import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from prepare_sources import prepare_source
from blocking import train_fastText, vectorize_instances,  build_LSH_tables, define_pw_matchings_to_perform
from pairwise_matching import get_index_mapping, build_similarity_matrix, compute_expected_cluster_size
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

### PHASE 3 -> PAIRWISE MATCHING ###
print("\n*** PHASE 3 -> PAIRWISE MATCHING ***")
instance_vectors = pd.read_csv(f'./datasets/{DATASET}/instance_vectors.csv', index_col=0, header=0)
index_mapping = get_index_mapping(instance_vectors)
similarity_matrix = build_similarity_matrix(instance_vectors, index_mapping, DATASET)
ecs = compute_expected_cluster_size(similarity_matrix, index_mapping)

### PHASE 4 -> CLUSTERING ###
print("\n*** PHASE 4 -> CLUSTERING ***")
llm = instantiate_llm()
rl_prompt = build_pw_prompt()

node_clusters = dict()
for idx in instances.index:
    node_clusters[idx] = set()

for k in tqdm(ecs.keys(), desc="Processing nodes"):
    if len(node_clusters[k]) == 0:
        node_id = index_mapping[k]
        neighbors = dict()
        for i, sim in enumerate(similarity_matrix[index_mapping[k]]):
            if sim > 0:
                neighbors[instances.index[i]] = sim
        
        neighbors = {k: v for k, v in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)}

        for n in neighbors:
            resp = query_llm(llm, rl_prompt, DATASET, instances, k, n)
            if 'MATCH' in resp:
                node_clusters[k].add(n)
                node_clusters[k].update(node_clusters[n])

                node_clusters[n].add(k)
                node_clusters[n].update(node_clusters[k])

with open(f'./datasets/{DATASET}/report.txt', 'w') as f:
    for k in node_clusters:
        f.write(f"Node {k} -> {node_clusters[k]}\n")