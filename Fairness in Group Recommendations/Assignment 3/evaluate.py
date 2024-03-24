def user_satisfaction(group_scores, user_scores):
    group_scores = {k: v for k, v in sorted(group_scores.items(), key=lambda item: item[1], reverse=True)}
    user_scores = {k: v for k, v in sorted(user_scores.items(), key=lambda item: item[1], reverse=True)}

    top_10_group = list(group_scores.items())[:10]
    top_10_user = list(user_scores.items())[:10]

    group_list_sat, user_list_sat = 0, 0
    for i in range(0, 10):
        user_list_sat += top_10_user[i][1]

        movie_recommended = top_10_group[i][0]
        group_list_sat += user_scores[movie_recommended]
    
    return (group_list_sat/user_list_sat)