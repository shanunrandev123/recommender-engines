import pandas as pd
import numpy as np

# Load the data

df = pd.read_csv(r'C:\Users\Asus\Downloads\archive (5)\rating.csv')
#
# print(df.head())
#
# print(df.userId.nunique())

df.userId = df.userId - 1

unique_movie_ids = set(df.movieId.values)

movie2idx = {}

count = 0

for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1


df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)

df = df.drop(columns=['timestamp'])

df.to_csv('C:/Users/Asus/Documents/recommender_engines/edited_rating.csv')