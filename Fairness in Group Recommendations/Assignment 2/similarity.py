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