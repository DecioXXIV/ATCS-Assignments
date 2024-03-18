def kendalltau_distance(y_true, y_pred):
    # This function calculates the Kendall Tau distance between two rankings
    y_pred = {k: v for k, v in sorted(y_pred.items(), key=lambda item: item[1], reverse=True)}
    y_true = {k: v for k, v in sorted(y_true.items(), key=lambda item: item[1], reverse=True)}

    y_true_filtered = {k: v for k,v in y_true.items() if k in y_pred.keys()}

    # Each movie, which is a "not-yet-seen" movie for all the users of the considered group, occupies a position in two rankings:
        # - the "true" ranking, which is the ranking of the movies according to the preferences of the current user.
        # - the "predicted" ranking, which is the ranking of the movies according to the preferences of the group.
    
    # To compute the fairness of the User-Based Collaborative Filtering, we calculate the number of discordant pairs of movies in the two rankings.

    pos2key_pred = {elem : i for i, elem in enumerate(y_pred.keys())}               # This dictionary contains pairs (movie_id, position_in_the_rank) in the "predicted" ranking.
    key2pos_true = {i : elem for i, elem in enumerate(y_true_filtered.keys())}      # This dictionary contains pairs (position_in_the_rank, movie_id) in the "true" ranking.

    discordant_pairs = 0

    for i in range(0, len(y_true_filtered)-1):
        true_pos_i = i                      
        elem_i = key2pos_true[i]                    # We retrieve the "movie_id" of the movie that occupies the "i-th" position in the "true" ranking.
        pred_pos_i = pos2key_pred[elem_i]           # We retrieve the "predicted_position" for the movie that occupies the "i-th" position in the "true" ranking.

        for j in range(i+1, len(y_true_filtered)):
            true_pos_j = j                  
            elem_j = key2pos_true[j]                # We retrieve the "movie_id" of the movie that occupies the "j-th" position in the "true" ranking.
            pred_pos_j = pos2key_pred[elem_j]       # We retrieve the "predicted_position" for the movie that occupies the "j-th" position in the "true" ranking.

            if (true_pos_i < true_pos_j) and (pred_pos_i > pred_pos_j):
                discordant_pairs += 1
            
            if (true_pos_i > true_pos_j) and (pred_pos_i < pred_pos_j):
                discordant_pairs += 1

    return discordant_pairs