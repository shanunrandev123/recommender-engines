import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sortedcontainers import SortedList

from datetime import datetime

with open (r'/user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)
    
    
with open (r'/movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)
    
with open (r'/usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)
    
    
with open (r'/usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)
    
    


# print(list(user2movie.keys()))


N = np.max(list(user2movie.keys())) + 1

m1 = np.max(list(movie2user.keys()))

m2 = np.max([m for (u,m), r in usermovie2rating_test.items()])

M = max(m1, m2) + 1


K = 25 #number of neighbors we would like to consider
limit = 5 # number of common movies users must have in common in order to consider
neighbors = [] # store neighbors in this list
averages = [] # each user's average rating
deviations = [] # each user's deviation

for i in range(N):
    
    #finding the 25 closest users to user i
    
    movie_i = user2movie[i]
    
    movie_i_set = set(movie_i)
    
    ratings_i = {movie : usermovie2rating[(i, movie)] for movie in movie_i}
    
    avg_i = np.mean(list(ratings_i.values()))
    
    dev_i = {movie : (rating - avg_i) for movie, rating in ratings_i.items()}
    
    dev_i_values = np.array(list(dev_i.values()))
    
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))
    
    averages.append(avg_i)
    
    deviations.append(dev_i)
    
    
    sl = SortedList()
    for j in range(N):
        if j != i:
            movie_j = user2movie[j]
            movie_j_set = set(movie_j)

            common_movies = (movie_i_set & movie_j_set)

            if len(common_movies) > limit:
                ratings_j = {movie : usermovie2rating[(j, movie)] for movie in movie_j}

                avg_j = np.mean(list(ratings_j.values()))

                dev_j = {movie : (rating - avg_j) for movie, rating in ratings_j.items()}

                dev_j_values = np.array(list(dev_j.values()))

                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)

                W_ij = numerator/(sigma_i * sigma_j)

                sl.add((-W_ij, j))

                if len(sl) > K:
                    del sl[-1]
    neighbors.append(sl)


    if i % 50 == 0:
        print(i)






#uses an user i and movie m to predict the rating of user i on movie m
def predict(i, m):
    
    numerator = 0
    denominator = 0
    
    for neg_W, j in neighbors[i]:
        # weight is stored as its negative
        # so the negative of the negative weight is the positive weight
        
        try:
            numerator += -neg_W + deviations[j][m]
            denominator += abs(neg_W)
            
        except KeyError:
            pass
        
        
    if denominator == 0:
        prediction = averages[i]
        
    else:
        prediction = numerator/denominator + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction)
    return prediction


train_predictions = []
train_targets = []

for (i,m), target in usermovie2rating.items():
    prediction = predict(i, m)

    train_predictions.append(prediction)
    train_targets.append(target)

test_predictions = []
test_targets = []

for (i,m), target in usermovie2rating_test.items():
    prediction = predict(i, m)

    test_predictions.append(prediction)
    test_targets.append(target)


def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)


print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))




            
            
    


