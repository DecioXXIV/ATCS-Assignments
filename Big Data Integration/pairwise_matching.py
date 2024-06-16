import numpy as np

def get_index_mapping(instance_vectors):
    mapping = dict()
    for i, idx in enumerate(instance_vectors.index):
        mapping[idx] = i
    
    return mapping

def build_similarity_matrix(instance_vectors, index_mapping, DATASET):
    print("Building Similarity Matrix...")
    similarity_matrix = np.zeros((len(instance_vectors), len(instance_vectors)))
    
    with open(f"./datasets/{DATASET}/pw_matchings_to_perform.csv", "r") as f:
        for line in f:
            line = line.replace('\n', '')
            i1, i2 = line.split(',')
            v1, v2 = instance_vectors.loc[i1].values, instance_vectors.loc[i2].values

            cos_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            similarity_matrix[index_mapping[i1], index_mapping[i2]] = cos_similarity
            similarity_matrix[index_mapping[i2], index_mapping[i1]] = cos_similarity
    
    print("Similarity Matrix Built!\n")
    return similarity_matrix

def compute_expected_cluster_size(similarity_matrix, index_mapping):
    print("Computing Expected Cluster Size...")
    ecs = dict()
    for key in index_mapping.keys():
        total_sim = np.sum(similarity_matrix[index_mapping[key]])
        ecs[key] = total_sim
    
    ecs = {k: v for k, v in sorted(ecs.items(), key=lambda item: item[1], reverse=True)}
    
    print("Expected Cluster Size Computed!")
    return ecs