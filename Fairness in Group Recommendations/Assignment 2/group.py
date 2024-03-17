import numpy as np

def average_aggregation(list_of_scores):
    key_set = list_of_scores[0].keys()
    for i in range(1, len(list_of_scores)):
        other_key_set = list_of_scores[i].keys()
        key_set = key_set & other_key_set
    
    avg_matrix = dict()
    for key in key_set:
        value = np.zeros(len(list_of_scores))
        for i in range(0, len(list_of_scores)):
            value[i] = list_of_scores[i][key]
        avg_matrix[key] = np.mean(value)
    
    return avg_matrix

def least_misery_aggregation(list_of_scores):
    key_set = list_of_scores[0].keys()
    for i in range(1, len(list_of_scores)):
        other_key_set = list_of_scores[i].keys()
        key_set = key_set & other_key_set
    
    lm_matrix = dict()
    for key in key_set:
        value = np.zeros(len(list_of_scores))
        for i in range(0, len(list_of_scores)):
            value[i] = list_of_scores[i][key]
        lm_matrix[key] = np.min(value)
    
    return lm_matrix

def group_recommendation(predictions, movie_map, max_recommendations):
    predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}
    predictions = dict(list(predictions.items())[:max_recommendations])

    recommendations = dict()
    for movie_id in predictions.keys():
        key = movie_map.loc[movie_id]['title']
        value = predictions[movie_id]
        recommendations[key] = value
    
    return recommendations