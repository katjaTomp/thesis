import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM,GMMHMM

# Omit warnings

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Pickled file names
def loadData(t, condition):
    if( (t == True) & (condition == 'both')):

        participants_data_1 = 'ppn1-downsampled.pkl'
        participants_data_2 = 'ppn2-downsampled.pkl'
        participants_data_3 = 'ppn3-downsampled.pkl'
        participants_data_4 = 'ppn4-downsampled.pkl'
        participants_data_5 = 'ppn5-downsampled.pkl'
        participants_data_6 = 'ppn6-downsampled.pkl'
        participants_data_7 = 'ppn7-downsampled.pkl'
        participants_data_9 = 'ppn9-downsampled.pkl'
        participants_data_10 = 'ppn10-downsampled.pkl'
        participants_data_11 = 'ppn11-downsampled.pkl'
        participants_data_12 = 'ppn12-downsampled.pkl'
        participants_data_13 = 'ppn13-downsampled.pkl'

         # Read the data

        frontal_1 = np.array(pd.read_pickle(participants_data_1)['FCz']).reshape(-1, 1)
        frontal_2 = np.array(pd.read_pickle(participants_data_2)['FCz']).reshape(-1, 1)
        frontal_3 = np.array(pd.read_pickle(participants_data_3)['FCz']).reshape(-1, 1)
        frontal_4 = np.array(pd.read_pickle(participants_data_4)['FCz']).reshape(-1, 1)
        frontal_5 = np.array(pd.read_pickle(participants_data_5)['FCz']).reshape(-1, 1)
        frontal_6 = np.array(pd.read_pickle(participants_data_6)['FCz']).reshape(-1, 1)
        frontal_7 = np.array(pd.read_pickle(participants_data_7)['FCz']).reshape(-1, 1)
        frontal_9 = np.array(pd.read_pickle(participants_data_9)['FCz']).reshape(-1, 1)
        frontal_10 = np.array(pd.read_pickle(participants_data_10)['FCz']).reshape(-1, 1)
        frontal_11 = np.array(pd.read_pickle(participants_data_11)['FCz']).reshape(-1, 1)
        frontal_12 = np.array(pd.read_pickle(participants_data_12)['FCz']).reshape(-1, 1)
        frontal_13 = np.array(pd.read_pickle(participants_data_13)['FCz']).reshape(-1, 1)

        frontal_final = np.concatenate([frontal_1, frontal_2, frontal_3, frontal_4, frontal_5, frontal_6, frontal_7, frontal_9, frontal_10, frontal_11, frontal_12,frontal_13 ])
        lengths = [len(frontal_1),len(frontal_2), len(frontal_3), len(frontal_4), len(frontal_5), len(frontal_6), len(frontal_7),
          len(frontal_9), len(frontal_10), len(frontal_11),len(frontal_12), len(frontal_13)]

        return [frontal_final, lengths]


    elif ((t == False) & (condition == 'both')):

        participants_data_1 = '1-downsampledikk.pkl'
        participants_data_2 = '2-downsampledikk.pkl'
        participants_data_3 = '3-downsampledikk.pkl'
        participants_data_4 = '4-downsampledikk.pkl'
        participants_data_5 = '5-downsampledikk.pkl'
        participants_data_6 = '6-downsampledikk.pkl'
        participants_data_7 = '7-downsampledikk.pkl'
        participants_data_9 = '9-downsampledikk.pkl'
        participants_data_10 = '10-downsampledikk.pkl'
        participants_data_11 = '11-downsampledikk.pkl'
        participants_data_12 = '12-downsampledikk.pkl'
        participants_data_13 = '13-downsampledikk.pkl'

        frontal_1 = np.array(pd.read_pickle(participants_data_1)['FCz']).reshape(-1, 1)
        frontal_2 = np.array(pd.read_pickle(participants_data_2)['FCz']).reshape(-1, 1)
        frontal_3 = np.array(pd.read_pickle(participants_data_3)['FCz']).reshape(-1, 1)
        frontal_4 = np.array(pd.read_pickle(participants_data_4)['FCz']).reshape(-1, 1)
        frontal_5 = np.array(pd.read_pickle(participants_data_5)['FCz']).reshape(-1, 1)
        frontal_6 = np.array(pd.read_pickle(participants_data_6)['FCz']).reshape(-1, 1)
        frontal_7 = np.array(pd.read_pickle(participants_data_7)['FCz']).reshape(-1, 1)
        frontal_9 = np.array(pd.read_pickle(participants_data_9)['FCz']).reshape(-1, 1)
        frontal_10 = np.array(pd.read_pickle(participants_data_10)['FCz']).reshape(-1, 1)
        frontal_11 = np.array(pd.read_pickle(participants_data_11)['FCz']).reshape(-1, 1)
        frontal_12 = np.array(pd.read_pickle(participants_data_12)['FCz']).reshape(-1, 1)
        frontal_13 = np.array(pd.read_pickle(participants_data_13)['FCz']).reshape(-1, 1)

        frontal_final = np.concatenate([frontal_1, frontal_2, frontal_3, frontal_4, frontal_5, frontal_6, frontal_7, frontal_9, frontal_10, frontal_11, frontal_12,frontal_13 ])
        lengths = [len(frontal_1),len(frontal_2), len(frontal_3), len(frontal_4), len(frontal_5), len(frontal_6), len(frontal_7),
          len(frontal_9), len(frontal_10), len(frontal_11),len(frontal_12), len(frontal_13)]

        return [frontal_final, lengths]
        #return [frontal_1, len(frontal_1)]

    elif condition == 'active':
        if t == False :
            participants_data_1 = '1-downsampledikk.pkl'
            participants_data_2 = '2-downsampledikk.pkl'
            participants_data_3 = '3-downsampledikk.pkl'
            participants_data_7 = '7-downsampledikk.pkl'
            participants_data_9 = '9-downsampledikk.pkl'
            participants_data_11 = '11-downsampledikk.pkl'
            participants_data_13 = '13-downsampledikk.pkl'

        else:
            participants_data_1 = 'ppn1-downsampled.pkl'
            participants_data_2 = 'ppn2-downsampled.pkl'
            participants_data_3 = 'ppn3-downsampled.pkl'
            participants_data_7 = 'ppn7-downsampled.pkl'
            participants_data_9 = 'ppn9-downsampled.pkl'
            participants_data_11 = 'ppn11-downsampled.pkl'
            participants_data_13 = 'ppn13-downsampled.pkl'


        frontal_final_active_1 = np.array(pd.read_pickle(participants_data_1)['FCz']).reshape(-1, 1)
        frontal_final_active_2 = np.array(pd.read_pickle(participants_data_2)['FCz']).reshape(-1, 1)
        frontal_final_active_3 = np.array(pd.read_pickle(participants_data_3)['FCz']).reshape(-1, 1)
        frontal_final_active_7 = np.array(pd.read_pickle(participants_data_7)['FCz']).reshape(-1, 1)
        frontal_final_active_9 = np.array(pd.read_pickle(participants_data_9)['FCz']).reshape(-1, 1)
        frontal_final_active_11 = np.array(pd.read_pickle(participants_data_11)['FCz']).reshape(-1, 1)
        frontal_final_active_13 = np.array(pd.read_pickle(participants_data_13)['FCz']).reshape(-1, 1)


        frontal_final_active = np.concatenate([frontal_final_active_1,frontal_final_active_2,frontal_final_active_3,frontal_final_active_7,frontal_final_active_9,frontal_final_active_11,frontal_final_active_13 ])

        lengths = [len(frontal_final_active_1), len(frontal_final_active_2), len(frontal_final_active_3), len(frontal_final_active_7),len(frontal_final_active_9), len(frontal_final_active_11),len(frontal_final_active_13 )]

        return [frontal_final_active, lengths]

    elif condition == 'passive':

        if t == False:

            participants_data_4 = '4-downsampledikk.pkl'
            participants_data_5 = '5-downsampledikk.pkl'
            participants_data_6 = '6-downsampledikk.pkl'
            participants_data_10 = '10-downsampledikk.pkl'
            participants_data_12 = '12-downsampledikk.pkl'
        else:

            participants_data_4 = 'ppn4-downsampled.pkl'
            participants_data_5 = 'ppn5-downsampled.pkl'
            participants_data_6 = 'ppn6-downsampled.pkl'
            participants_data_10 = 'ppn10-downsampled.pkl'
            participants_data_12 = 'ppn12-downsampled.pkl'


        frontal_final_passive_4 = np.array(pd.read_pickle(participants_data_4)['FCz']).reshape(-1, 1)
        frontal_final_passive_5 = np.array(pd.read_pickle(participants_data_5)['FCz']).reshape(-1, 1)
        frontal_final_passive_6 = np.array(pd.read_pickle(participants_data_6)['FCz']).reshape(-1, 1)
        frontal_final_passive_10 = np.array(pd.read_pickle(participants_data_10)['FCz']).reshape(-1, 1)
        frontal_final_passive_12 = np.array(pd.read_pickle(participants_data_12)['FCz']).reshape(-1, 1)

        frontal_final_passive = np.concatenate([frontal_final_passive_4, frontal_final_passive_5, frontal_final_passive_6, frontal_final_passive_10, frontal_final_passive_12])
        lengths = [len(frontal_final_passive_4), len(frontal_final_passive_5), len(frontal_final_passive_6), len(frontal_final_passive_10), len(frontal_final_passive_12)]

        return [frontal_final_passive, lengths]



def trainModel(X,lengths,states):
    model = GaussianHMM(n_components=states, covariance_type="diag", n_iter=1000).fit(X, lengths)

    print(model.predict(X))
    print(model.monitor_.converged)
    print(model.monitor_)
    print(model.score(X, lengths))
    return model


def trainModelGMM(X, lengths, states, num_gaus):

    model = GMMHMM(n_components=states, n_mix=num_gaus,n_iter=1000,verbose=True).fit(X,lengths)

    print('Mixture Models + HMM')
    print(model.predict(X))
    print(model.monitor_.converged)
    print(model.monitor_)
    print(model.score(X, lengths))


def saveModel(model):
     from sklearn.externals import joblib
     joblib.dump(model, "model.pkl")



data = loadData(False, "both")
#print (data[0])
frontal_final = data[0]
lengths = data[1]
#print('length',sum(lengths))

import math

mean = np.mean(frontal_final)
std = np.std(frontal_final)
## Train models
#print(mean,std)
model = trainModel(frontal_final, lengths, 6)
saveModel(model)


#for i in range(2, 12):
#    print(" The model was trained using ", i, "states")
#    trainModel(frontal_final, lengths, i)


#trainModelGMM(frontal_final,lengths,3,3)
from sklearn.externals import joblib
from hmmlearn.hmm import GaussianHMM

model = joblib.load( "model.pkl")

#print(model.score(frontal_final,lengths))
predictions = model.predict(frontal_final)
A = model.transmat_
means = model.means_
pi = model.startprob_
B = model.predict_proba(frontal_final)
covars = model.covars_
print (A, means, pi, covars,B)



### Plot the results
import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from matplotlib import cm, pyplot as plt


X = frontal_final

Y = predictions
j = 0
print j
timepoint = 0

for i in Y:

    if i != j:
        print timepoint ,":",i
        j=i
    timepoint +=1

plt.eventplot(Y)
plt.show()

