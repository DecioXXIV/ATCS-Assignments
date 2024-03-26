import numpy as np

def pearson_similarity(matrix, input_user, other_user):
    """
    Compute the Pearson Similarity between two users.
    
    Parameters:
    - X (pandas.DataFrame): The User-Item rating matrix.
    - input_user (int): The ID of the Input User.
    - other_user (int): The ID of the Other User.
    """
    input_mean = matrix.loc[input_user].mean()
    other_mean = matrix.loc[other_user].mean()

    # Select only the matrix's columns (movies) that have non-missing values for both the Input User and the Other User
    # What does "~" do? It inverts the boolean value computed for the following condition
    common_columns = matrix.columns[(~matrix.loc[input_user].isnull()) & (~matrix.loc[other_user].isnull())]

    # Retrieve the co-ratings
    input_ratings = matrix.loc[input_user, common_columns]
    other_ratings = matrix.loc[other_user, common_columns]

    # Similarity Computation
    numerator = np.sum((input_ratings - input_mean) * (other_ratings - other_mean))
    denominator = np.sqrt(np.sum((input_ratings - input_mean)**2)) * np.sqrt(np.sum((other_ratings - other_mean)**2))

    if denominator == 0:
        return 0
    else:
        return numerator / denominator