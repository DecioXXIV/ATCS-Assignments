import pandas as pd
import numpy as np
from datetime import datetime
from preprocessing import get_matrix, get_movie_map
from similarity import pearson_similarity, weighted_pearson_similarity, norm_weights
from predict import recommend_movies

MAX_NEIGHBORS = 10
MAX_RECOMMENDATIONS = 10

ratings = pd.read_csv('./datasets/ratings.csv')
movies = pd.read_csv('./datasets/movies.csv')

start = datetime.now()
print("START OF DATASET PREPROCESSING")
matrix = get_matrix(ratings, movies)
movie_map = get_movie_map(movies)
print("End of Dataset Preprocessing, elapsed time: " + str(datetime.now() - start) + "\n")

input_user = np.random.randint(low=1, high=matrix.shape[0]+1)
print(f'Input user: {input_user}')
other_users = [u for u in matrix.index.tolist() if u != input_user]

print("### ################################### ###")
print("### FIRST EXECUTION: Pearson Similarity ###")
print("### ################################### ###\n")

start = datetime.now()

print("### First Step: Similarity Computation")
similarities = dict()
for u in other_users:
    similarities[u] = pearson_similarity(matrix, input_user, u)

similarities = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}
similarities = dict(list(similarities.items())[:MAX_NEIGHBORS])

with open(f'./base_similarities_{input_user}.txt', 'w') as f:
    f.write(str(similarities))
print("Similarities computed!\n")

print("### Second Step: Recommendations")
recommendations = recommend_movies(matrix, input_user, similarities, MAX_RECOMMENDATIONS, movie_map)
with open(f'./base_recommendations_{input_user}.txt', 'w') as f:
    f.write(str(recommendations))

print("Recommendations:", recommendations)
print("\nEnd of FIRST EXECUTION, elapsed time: " + str(datetime.now() - start) + "\n")

print("### ############################################# ###")
print("### SECOND EXECUTION: Weighted Pearson Similarity ###")
print("### ############################################# ###\n")

start = datetime.now()

print("### First Step: Similarity Computation")

norm_scores = norm_weights(input_user, matrix)

weighted_similarities = dict()
for u in other_users:
    weighted_similarities[u] = weighted_pearson_similarity(matrix, input_user, u, norm_scores.get(u))

weighted_similarities = {k: v for k, v in sorted(weighted_similarities.items(), key=lambda item: item[1], reverse=True)}
weighted_similarities = dict(list(weighted_similarities.items())[:MAX_NEIGHBORS])

with open(f'./weighted_similarities_{input_user}.txt', 'w') as f:
    f.write(str(weighted_similarities))
print("Similarities computed!\n")

print("### Second Step: Recommendations")
weighted_recommendations = recommend_movies(matrix, input_user, weighted_similarities, MAX_RECOMMENDATIONS, movie_map)

print("Recommendations:", weighted_recommendations)

with open(f'./weighted_recommendations_{input_user}.txt', 'w') as f:
    f.write(str(weighted_recommendations))

print("\nEnd of SECOND EXECUTION, elapsed time: " + str(datetime.now() - start) + "\n")