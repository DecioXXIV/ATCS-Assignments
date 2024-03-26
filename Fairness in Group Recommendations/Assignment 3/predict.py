import numpy as np

def prediction_function(matrix, input_user, similarities, max_recommendations):
    """
    Apply the User-Based Collaborative Filtering Algorithm to predict the ratings for the Input User.

    Parameters:
    - matrix (pandas.DataFrame): The User-Item rating matrix.
    - input_user (int): The ID of the Input User.
    - similarities (dict): dictionary of similarities between the Input User and another User -> pairs: (other_user, similarity)
    - max_recommendations (int): The maximum number of recommendations to return.
    """
    input_mean = matrix.loc[input_user].mean()

    not_seen_movies = matrix.columns[matrix.loc[input_user].isna()]
    scores = dict()

    for movie in not_seen_movies:
        numerator, denominator = 0, 0
        for (other_user, similarity) in similarities.items():
            if not np.isnan(matrix.loc[other_user, movie]):
                other_mean = matrix.loc[other_user].mean()
                numerator += similarity * (matrix.loc[other_user, movie] - other_mean)
                denominator += np.abs(similarity)
        
        scores[movie] = input_mean + (numerator / denominator) if denominator != 0 else 0
    
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    scores = dict(list(scores.items())[:max_recommendations])

    return scores