def user_satisfaction(group_scores, user_scores, max_recommendations):
    """
    Compute the User Satisfaction for the Group Recommendations.

    Parameters:
    - group_scores (dict): A dictionary containing the Scores predicted for the Group.
    - user_scores (dict): A dictionary containing the Scores predicted for the User.
    - max_recommendations (int): The maximum number of recommendations to consider for the evaluation.
    """
    group_scores = {k: v for k, v in sorted(group_scores.items(), key=lambda item: item[1], reverse=True)}
    user_scores = {k: v for k, v in sorted(user_scores.items(), key=lambda item: item[1], reverse=True)}

    top_10_group = list(group_scores.items())[:max_recommendations]
    top_10_user = list(user_scores.items())[:max_recommendations]

    group_list_sat, user_list_sat = 0, 0
    for i in range(0, max_recommendations):
        user_list_sat += top_10_user[i][1]

        movie_recommended = top_10_group[i][0]
        group_list_sat += user_scores[movie_recommended]
    
    # The "User Satisfaction" is the following ratio:
        # num: Sum of the Scores related to the Group Recommendations 
        # den: Sum of the Scores related to the User Recommendations
    return (group_list_sat/user_list_sat)