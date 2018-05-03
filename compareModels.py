from sklearn.externals import joblib
from hmmlearn.hmm import GaussianHMM
import numpy as np
import csv

import pickle

state_change = pickle.load(open('states_all.pkl', 'r'))
states = np.reshape(state_change,(-1,1))
print (states)

with open('states_all.csv', 'wb') as f:
    writer = csv.writer(f)

    writer.writerows(states)

