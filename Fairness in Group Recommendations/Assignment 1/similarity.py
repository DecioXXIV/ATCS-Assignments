import numpy as np

def pearson_similarity(matrix, input_user, other_user):
    # Implemetation of the "Pearson Similarity" formula
    input_coratings, other_coratings = list(), list()

    input_mean, other_mean = matrix.loc[input_user].mean(), matrix.loc[other_user].mean()

    for i in matrix.columns.tolist():
        if not np.isnan(matrix.loc[input_user, i]) and not np.isnan(matrix.loc[other_user, i]):
            input_coratings.append(matrix.loc[input_user, i])
            other_coratings.append(matrix.loc[other_user, i])
    
    input_coratings, other_coratings = np.array(input_coratings), np.array(other_coratings)

    numerator = np.sum((input_coratings - input_mean) * (other_coratings - other_mean))
    denominator = np.sqrt(np.sum((input_coratings - input_mean)**2)) * np.sqrt(np.sum((other_coratings - other_mean)**2))

    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def weighted_pearson_similarity(matrix, input_user, other_user, norm_coeff):
    # Implemetation of the "Pearson Similarity" formula with a normalization coefficient: the "norm_coeff" gives more importance to the preferences of the neighbors with more co-rated movies with the input user.
    input_coratings, other_coratings = list(), list()

    input_mean, other_mean = matrix.loc[input_user].mean(), matrix.loc[other_user].mean()

    for i in matrix.columns.tolist():
        if not np.isnan(matrix.loc[input_user, i]) and not np.isnan(matrix.loc[other_user, i]):
            input_coratings.append(matrix.loc[input_user, i])
            other_coratings.append(matrix.loc[other_user, i])
    
    input_coratings, other_coratings = np.array(input_coratings), np.array(other_coratings)

    numerator = np.sum((input_coratings - input_mean) * (other_coratings - other_mean))
    denominator = np.sqrt(np.sum((input_coratings - input_mean)**2)) * np.sqrt(np.sum((other_coratings - other_mean)**2))

    if denominator == 0:
        return 0
    else:
        return norm_coeff * numerator / denominator

def norm_weights(input_user, matrix):
    weights = dict()
    other_users = [u for u in matrix.index.tolist() if u != input_user]
    coratings = list()

    for u in other_users:
        coratings.append(np.sum(matrix.loc[input_user].notna() & matrix.loc[u].notna()))
    
    coratings = np.array(coratings)
    tot_coratings = np.sum(coratings)

    scores = coratings / tot_coratings

    for u, s in zip(other_users, scores):
        weights[u] = s
    
    weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1], reverse=True)}

    # The preferences for the neighbors with more co-rated movies with the input user are more important
    return weights