import os
import itertools
import pandas as pd
import numpy as np
import fasttext as fastText
from tqdm import tqdm

def train_fastText(instances, LS_DIM, DATASET):
    with open(f"./datasets/{DATASET}/instances_refined.txt", "w") as f:
        for idx in instances.index:
            line = ""
            for col in instances.columns:
                if instances.at[idx, col] != 'nan':
                    line += instances.at[idx, col] + " "
            f.write(line + "\n")
    
    model = fastText.train_unsupervised(f"./datasets/{DATASET}/instances_refined.txt", model='cbow', lr=0.07, dim=LS_DIM, ws=2, epoch=500, minCount=1, wordNgrams=3, thread=os.cpu_count(), verbose=2)

    return model

def vectorize_instances(instances, ft_model, word_weights, DATASET):
    instance_vectors = pd.DataFrame(columns=range(0, ft_model.get_dimension()), index=instances.index)
    for idx in tqdm(instances.index, desc="Vectorizing Instances"):
        attribute_vectors = list()
        for col in instances.columns:
            text = instances.at[idx, col]
            if text != 'nan' and text != '':
                vector = np.zeros(ft_model.get_dimension())
                tokens = text.split()
                total_weight = 0
                for t in tokens:
                    weight = word_weights.at[t, 'weight']
                    total_weight += weight
                    vector += ft_model.get_word_vector(t) * weight
                vector /= total_weight
                attribute_vectors.append(vector)
        
        final_vector = np.zeros(ft_model.get_dimension())
        for v in attribute_vectors:
            final_vector += v
        final_vector /= len(attribute_vectors)
        instance_vectors.loc[idx] = final_vector
    
    instance_vectors.to_csv(f"./datasets/{DATASET}/instance_vectors.csv", header=True, index=True)
    return instance_vectors

def generate_buckets(K):
    values = [-1, 1]
    vectors = list(itertools.product(values, repeat=K))
    return vectors

def generate_random_hyperplanes(K, LS_DIM):
    hyperplanes = np.random.randn(K, LS_DIM)
    norms = np.linalg.norm(hyperplanes, axis=1, keepdims=True)
    hyperplanes = hyperplanes / norms
    return hyperplanes

def build_LSH_tables(instance_vectors, L, K, LS_DIM):
    tables = list()
    for i in range(L):
        table = dict()
        bucket_keys = generate_buckets(K)
        buckets = dict()
        for key in bucket_keys:
            buckets[key] = list()
        
        table['hyperplanes'] = generate_random_hyperplanes(K, LS_DIM)
        table['buckets'] = buckets
        tables.append(table)
    
    for idx in tqdm(instance_vectors.index, desc="Generating LSH Tables"):
        vector = instance_vectors.loc[idx]
        for table in tables:
            hash_key = tuple(np.sign(np.dot(table['hyperplanes'], vector)))
            table['buckets'][hash_key].append(idx)
    
    return tables

def define_pw_matchings_to_perform(tables, DATASET):
    total_comparisons = set()
    for table in tables:
        buckets = table['buckets']
        for instance_list in buckets.values():
            total_comparisons.update(itertools.combinations(instance_list, 2))
    
    with open(f"./datasets/{DATASET}/pw_matchings_to_perform.csv", "w") as f:
        to_skip = set()
        for pair in total_comparisons:
            i1, i2 = pair
            if (i2, i1) in total_comparisons:
                to_skip.add((i2, i1))
            
            i1_source, i2_source = i1[0], i2[0]
            if (i1, i2) not in to_skip and i1_source != i2_source:
                f.write(f"{i1},{i2}\n")