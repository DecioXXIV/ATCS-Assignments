import os
import random
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

ratings = pd.read_csv("./datasets/ratings.csv")
movies = pd.read_csv("./datasets/movies.csv")
NUM_THREADS = os.cpu_count()
ratings_dict = dict()

def get_movie_map():
    movie_map = pd.DataFrame(data=movies['title'].values, index=movies['movieId'].values, columns=['title'])
    return movie_map

def split_dataset(num_iterations, seed):
    """
    Split the Dataset into several parts.
    The first part (50% of the total Dataset) is used as the starting Dataset.
    The other parts (num = num_iterations-1) are used to enrich the Dataset after each iteration.

    Parameters:
    - num_iterations (int): The number of iterations to be performed.
    - seed (int): The seed for the random shuffling of the Dataset.
    """
    indices_split = np.array_split(ratings.index, NUM_THREADS)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        for i, indices in enumerate(indices_split):
            df_subset = ratings.loc[indices]
            executor.submit(process_subdf, df_subset)
    
    list_of_ratings = list(ratings_dict.items())
    # The "list_of_ratings" is a list of tuples: ((user, movie), rating)

    random.seed(seed)
    random.shuffle(list_of_ratings)

    first_split = int(len(list_of_ratings)/2)
    first_half = list_of_ratings[:first_split]

    second_half = list_of_ratings[first_split:]
    chunk_split = int(len(second_half)/(num_iterations-1))+1

    chunks = list()
    for i in range(num_iterations-1):
        chunks.append(second_half[i*chunk_split:(i+1)*chunk_split])
    
    return first_half, chunks

def process_subdf(df):
    for index, row in df.iterrows():
        user = int(row['userId'])
        movie = int(row['movieId'])
        rating = row['rating']

        ratings_dict[(user, movie)] = rating
    
def get_starting_dataset(first_half):
    """
    Create the starting Dataset as a pandas.DataFrame

    Parameters:
    - first_half (list): The first half of the Dataset, containing the Ratings for the Movies.
    Each element is a tuple: ((user, movie), rating)
    """
    user_ids = ratings['userId'].unique().tolist()
    movie_ids = movies['movieId'].unique().tolist()

    matrix = pd.DataFrame(index=user_ids, columns=movie_ids, dtype=np.float32)

    for pair in first_half:
        user, movie = pair[0][0], pair[0][1]
        rating = pair[1]
        matrix.at[user, movie] = rating
    
    return matrix

def enrich_dataset(matrix, chunk):
    """
    Enriches the Dataset with the Ratings from the current chunk.

    Parameters:
    - matrix (pandas.DataFrame): The current Dataset.
    - chunk (list): The chunk containing the Ratings for the Movies.
    Each element of "chunk" is a tuple: ((user, movie), rating)
    """
    for pair in chunk:
        user, movie = pair[0][0], pair[0][1]
        rating = pair[1]
        matrix.at[user, movie] = rating

    return matrix