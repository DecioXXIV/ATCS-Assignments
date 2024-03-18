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

def weighted_disagreements_aggregation(list_of_scores):
    # The aggregation of Scores is performed only for the movies that are "not-yet-seen" for all the users of the considered group
    # Each dictionary in the "list_of_scores" contains the scores of the movies that are "not-yet-seen" for the user
    # The "key_set" is the intersection of the keys of all the dictionaries in the "list_of_scores"
    
    key_set = list_of_scores[0].keys()
    for i in range(1, len(list_of_scores)):
        other_key_set = list_of_scores[i].keys()
        key_set = key_set & other_key_set
    
    wd_matrix = dict()

    # The aggregated Score for each movie is obtained by a weighted average of the (predicted) scores for that movie
    # In the weighted average, the weights correspond to the "pairwise disagreement" between the scores for that movie given by two users.
    
    indexes = list(range(0, len(list_of_scores)))
    pairs = list(combinations(indexes, 2))

    for key in key_set:
        numerator, denominator = 0, 0
        for pair in pairs:
            i,j = pair
            # The "pairwise disagreement" between the scores for the current movie is calculated as "1" plus the absolute difference between the scores
            disagreement = 1 + np.abs(list_of_scores[i][key] - list_of_scores[j][key])

            # The pairwise disagreement is used as a multiplicative weight for the average score given to that movie by the two users of the pair.
            numerator += disagreement * np.mean((list_of_scores[i][key], list_of_scores[j][key]))
            denominator += disagreement
        
        wd_matrix[key] = numerator / denominator
    
    return wd_matrix

def group_recommendation(predictions, movie_map, max_recommendations):
    predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}
    predictions = dict(list(predictions.items())[:max_recommendations])

    recommendations = dict()
    for movie_id in predictions.keys():
        key = movie_map.loc[movie_id]['title']
        value = predictions[movie_id]
        recommendations[key] = value
    
    return recommendations