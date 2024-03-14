import numpy as np
import pandas as pd

def get_matrix(ratings, movies):
    # Each row in "ratings" stores ('userId', 'movieId', 'rating') -> we can retrieve the sed of user_ids
    user_ids = ratings['userId'].unique().tolist()

    # Each row in "movies" stores ('movieId', 'title', 'genres') -> we can retrieve the set of movie_ids
    movie_ids = movies['movieId'].unique().tolist()

    # The matrix stores all the ratings given by the users to the watched movies
    matrix = pd.DataFrame(index=user_ids, columns=movie_ids, dtype=np.float32)

    # For each row in "ratings", we retrieve a rating given by a user to a movie
    # We can populate the matrix (users, movies) with the ratings -> Be Careful! The matrix is very sparse!
    for i in range(0, len(ratings)):
        row = ratings.iloc[i]
        matrix.at[row['userId'], row['movieId']] = row['rating']
    
    # Matrix Control: we check if the matrix is correctly populated
    for i in matrix.index.tolist():
        assert matrix.loc[i].count() == len(ratings[ratings['userId'] == i])

    return matrix

def get_movie_map(movies):
    movie_map = pd.DataFrame(columns=['movieId', 'title'], dtype=str)

    # For each row in "movies", we retrieve the couple (movie_id, movie_title)...
    for i in range(0, len(movies)):
        row = movies.iloc[i]
        movie_id, movie_title = row['movieId'], row['title']

        # ...and we store it in the DataFrame
        movie_map.at[i, 'movieId'] = movie_id
        movie_map.at[i, 'title'] = movie_title
    
    movie_map.set_index('movieId', inplace=True)
    
    # Movie Map Control: we check if the movie_map is correctly populated
    assert len(movie_map) == len(movies)

    return movie_map
