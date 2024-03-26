import numpy as np
from itertools import combinations

def average_aggregation(list_of_scores):
    """
    Obtain the Group Ratings by averaging the Scores predicted for the Users in the Group.
    The aggregation of Scores is performed only for the movies that are "not-yet-seen" for all the users of the considered group

    Parameters:
    - list_of_scores (list): A list of dictionaries containing the Scores predicted for each User in the Group.
    Each dictionary in the "list_of_scores" contains the scores of the movies that are "not-yet-seen" for the user
    """
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
    """
    Obtain the Group Ratings by applying the Least Misery Aggregation to the Scores predicted for the Users in the Group.
    The aggregation of Scores is performed only for the movies that are "not-yet-seen" for all the users of the considered group

    Parameters:
    - list_of_scores (list): A list of dictionaries containing the Scores predicted for each User in the Group.
    Each dictionary in the "list_of_scores" contains the scores of the movies that are "not-yet-seen" for the user
    """
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
    """
    Obtain the Group Ratings by applying an Hybrid Aggregation to the Scores predicted for the Users in the Group.
    The aggregation of Scores is performed only for the movies that are "not-yet-seen" for all the users of the considered group

    Parameters:
    - list_of_scores (list): A list of dictionaries containing the Scores predicted for each User in the Group.
    Each dictionary in the "list_of_scores" contains the scores of the movies that are "not-yet-seen" for the user
    - alpha (float): the coefficient that balances the Hybrid Aggregation (Hybrid = Average Aggregation + Least Misery Aggregation)
    """
    # The "key_set" is the intersection of the keys of all the dictionaries in the "list_of_scores"
    key_set = list_of_scores[0].keys()
    for i in range(1, len(list_of_scores)):
        other_key_set = list_of_scores[i].keys()
        key_set = key_set & other_key_set
    
    alpha_hybrid_matrix = dict()

    # The aggregated Score for each movie is obtained by combining the Average Aggregation and the Least Misery Aggregation
    # The combination is balanced by the value of "alpha"
        # alpha = 0 -> aggregation = Average Aggregation
        # alpha = 1 -> aggregation = Least Misery Aggregation
    for key in key_set:
        value = np.zeros(len(list_of_scores))
        for i in range(0, len(list_of_scores)):
            value[i] = list_of_scores[i][key]
        
        mean = np.mean(value)
        min = np.min(value)

        alpha_hybrid_matrix[key] = (1-alpha)*mean + alpha*min
    
    return alpha_hybrid_matrix

def update_alpha_mom_disagreements(satisfactions):
    """
    Obtain the new value for the "alpha" parameter by considering the Disagreements between the Satisfactions of the Users in the Group.

    Parameters:
    - satisfactions (pandas.Series): The Satisfactions of the Users in the Group, computed for the current iteration.
    """
    values = satisfactions.values

    value_pairs = list(combinations(values, 2))  # Pairs: (Satisfaction User_i, Satisfaction User_j)
    disagreements = np.zeros(len(value_pairs))

    for i, pair in enumerate(value_pairs):
        # disagreement(User_i, User_j) = |Satisfaction User_i - Satisfaction User_j|
        disagreements[i] = np.abs(pair[0] - pair[1])
    

    disagreement_pairs = list(combinations(disagreements, 2))   # Pairs: (Disagreement UserPair_k, Disagreement UserPair_l)
    averages = np.zeros(len(values))
    for i, pair in enumerate(disagreement_pairs):
        averages[i] = np.mean(pair) # For each pair of disagreements, we compute the average
    
    # The new value for "alpha" is the median of the averages
    return np.median(averages)

def group_recommendation(predictions, movie_map, max_recommendations):
    """
    Obtain the final Recommendations for the Group by selecting the top "max_recommendations" movies from the Predictions.

    Parameters:
    - predictions (dict): A dictionary containing the Predictions (ratings) for the Movies in the Catalog.
    - movie_map (pandas.DataFrame): A DataFrame containing the mapping between the Movie IDs and the Movie Titles.
    - max_recommendations (int): The number of Recommendations to be provided to the Group.
    """
    predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}
    predictions = dict(list(predictions.items())[:max_recommendations])

    recommendations = dict()
    for movie_id in predictions.keys():
        key = movie_map.loc[movie_id]['title']
        value = predictions[movie_id]
        recommendations[key] = value
    
    return recommendations