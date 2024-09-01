import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle

from datetime import datetime

with open (r'/user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)
    
    
with open (r'/movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)
    
with open (r'/usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)
    
    
with open (r'/usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)
    
    
N = np.max(list(user2movie.keys())) + 1

m1 = np.max([m for (u,m), r in usermovie2rating.items()])

m2 = np.max([m for (u,m), r in usermovie2rating_test.items()])

M = max(m1, m2) + 1

print("N:", N, "M:", M)



K = 10 #latent dimesnionality
W = np.random.randn(N,K)
b = np.zeros(N)

U = np.random.randn(M,K)
c = np.zeros(M)

mu = np.mean(list(usermovie2rating.values()))

# prediction[i, j] -> w[i].dot(U[j]) + b[i] + c.T[j] + mu

def get_loss(d):
    N = float(len(d))
    sse = 0
    for k, r in d.items():
        
        i, j = K
        
        p = W[i].dot(U[j]) + b[i] + c[j] + mu
        sse += (p - r)* (p - r)
    return sse/N



        
    