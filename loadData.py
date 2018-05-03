import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM,GMMHMM

from xlrd import open_workbook

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

        #return [frontal_final, lengths]
        return [frontal_1, len(frontal_1)]

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
     joblib.dump(model, "model_9_200.pkl")



data = loadData(False, "both")

frontal_final = data[0]
lengths = data[1]

## Train models
#model = trainModel(frontal_final, lengths, 9)
#saveModel(model)



#trainModelGMM(frontal_final,lengths,3,3)
from sklearn.externals import joblib
from hmmlearn.hmm import GaussianHMM
import pickle

model = joblib.load( "model_6_200.pkl")
events = pickle.load(open("ppn13_events","r"))
print (events)


#print(model.score(frontal_final,lengths))
predictions = model.predict(frontal_final)
A = model.transmat_
means = model.means_
pi = model.startprob_
B = model.predict_proba(frontal_final)
covars = model.covars_
#print (A, means, pi, covars,B)
print (covars)
state_0 = 5
states_change = []
states_change.append(state_0)

for prediction in predictions:
    if prediction != state_0:
     #print prediction
     state_0 = prediction
     states_change.append(state_0)

pickle.dump(predictions, open('states_all.pkl','w'))



#print(events)
#print(predictions)
### Plot the results
import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from matplotlib import cm, pyplot as plt


X = frontal_final
Y = predictions
print (len(Y))
down_x = []
down_time = []
j = 0
#print j
timepoint = 0
event_0 = ['65304']
#print event_0
for state, event in zip(Y,events):

    if event_0 != event:
        if event == ['65294']:
#            print event,":",state
            down_x.append(state)
            down_time.append(timepoint)
        #j=state
        event_0 = event
    timepoint +=1

plt.plot(down_x)
plt.show()


## number of states per event

event_codes = ['65304','65306','65308','65281','65294','65298','65296','65284','65286','65288']


def getFrequencies(event_1):
    zeros = 0
    ones = 0
    twos = 0
    threes = 0
    fours = 0
    fives = 0

    for state,event in zip(Y,events):
        if event == event_1:
            if state == 0:
                zeros +=1
            elif state == 1:
                ones +=1
            elif state == 2:
                twos +=1
            elif state == 3:
                threes +=1
            elif state == 4:
                fours +=1
            else:
                fives +=1


 #   print ('event:',event_1)
 #   print (zeros, ones, twos, threes, fours, fives)


for ev in event_codes:
    getFrequencies(ev)

event_0 = 0
state_0 = 0
states = []
states_seq = []
i = 0
for event, state in zip(events,predictions):
    if event_0 == event:
        if state_0 != state:
            states.append(state)
            state_0 = state
    else :
        seq  = -1
        for st in states:
            seq =str(seq) + ',' +str(st)
        event_0 = event
       # print seq
        states=[]

        i +=1
#print (states_seq)
