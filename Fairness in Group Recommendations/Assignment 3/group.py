import numpy as np
from itertools import combinations

def average_aggregation(list_of_scores):
    # The aggregation of Scores is performed only for the movies that are "not-yet-seen" for all the users of the considered group
    # Each dictionary in the "list_of_scores" contains the scores of the movies that are "not-yet-seen" for the user
    # The "key_set" is the intersection of the keys of all the dictionaries in the "list_of_scores"
    
    key_set = list_of_scores[0].keys()
    for i in range(1, len(list_of_scores)):
        other_key_set = list_of_scores[i].keys()
        key_set = key_set & other_key_set
    
    avg_matrix = dict()

    # The aggregated Score for each movie is the average of the (predicted) scores for that movie
    for key in key_set:
        value = np.zeros(len(list_of_scores))
        for i in range(0, len(list_of_scores)):
            value[i] = list_of_scores[i][key]
        avg_matrix[key] = np.mean(value)
    
    return avg_matrix

def least_misery_aggregation(list_of_scores):
    # The aggregation of Scores is performed only for the movies that are "not-yet-seen" for all the users of the considered group
    # Each dictionary in the "list_of_scores" contains the scores of the movies that are "not-yet-seen" for the user
    # The "key_set" is the intersection of the keys of all the dictionaries in the "list_of_scores"

    key_set = list_of_scores[0].keys()
    for i in range(1, len(list_of_scores)):
        other_key_set = list_of_scores[i].keys()
        key_set = key_set & other_key_set
    
    lm_matrix = dict()

    # The aggregated Score for each movie is the minimum of the (predicted) scores for that movie
    for key in key_set:
        value = np.zeros(len(list_of_scores))
        for i in range(0, len(list_of_scores)):
            value[i] = list_of_scores[i][key]
        lm_matrix[key] = np.min(value)
    
    return lm_matrix

def alpha_hybrid_aggregation(alpha, list_of_scores):
    key_set = list_of_scores[0].keys()
    for i in range(1, len(list_of_scores)):
        other_key_set = list_of_scores[i].keys()
        key_set = key_set & other_key_set
    
    alpha_hybrid_matrix = dict()

    for key in key_set:
        value = np.zeros(len(list_of_scores))
        for i in range(0, len(list_of_scores)):
            value[i] = list_of_scores[i][key]
        
        mean = np.mean(value)
        min = np.min(value)

        alpha_hybrid_matrix[key] = (1-alpha)*mean + alpha*min
    
    return alpha_hybrid_matrix

def group_recommendation(predictions, movie_map, max_recommendations):
    predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}
    predictions = dict(list(predictions.items())[:max_recommendations])

    recommendations = dict()
    for movie_id in predictions.keys():
        key = movie_map.loc[movie_id]['title']
        value = predictions[movie_id]
        recommendations[key] = value
    
    return recommendations