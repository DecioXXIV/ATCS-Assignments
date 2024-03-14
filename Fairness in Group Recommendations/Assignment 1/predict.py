import numpy as np

def prediction_function(matrix, input_user, similarities, max_recommendations):
    input_mean = matrix.loc[input_user].mean()

    not_seen_movies = matrix.columns[matrix.loc[input_user].isna()]
    scores = dict()

    for movie in not_seen_movies:
        numerator, denominator = 0, 0
        for (other_user, similarity) in similarities.items():
            if not np.isnan(matrix.loc[other_user, movie]):
                other_mean = matrix.loc[other_user].mean()
                numerator += similarity * (matrix.loc[other_user, movie] - other_mean)
                denominator += similarity
        
        scores[movie] = input_mean + (numerator / denominator) if denominator != 0 else 0
    
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    scores = dict(list(scores.items())[:max_recommendations])

    return scores

def recommend_movies(matrix, input_user, similarities, max_recommendations, movie_map):
    # Recommendations for the Input User: for each recommended movie, we also store the predicted rating
    predictions = prediction_function(matrix, input_user, similarities, max_recommendations)

    recommendations = dict()
    for movie_id in predictions.keys():
        key = movie_map.loc[movie_id]['title']
        value = predictions[movie_id]
        recommendations[key] = value
    
    return recommendations