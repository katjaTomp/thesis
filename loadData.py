import pandas as pd
import numpy as np

from sklearn import cluster
from hmmlearn.hmm import GaussianHMM,GMMHMM

#from xlrd import open_workbook

# Omit warnings

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Pickled file names
def loadData(t, condition):

    frontal = []
    if ((t == False) & (condition == 'both')):

        participants_data_1 = 'noEOG/1-downsampledikk.pkl'
        participants_data_2 = 'noEOG/2-downsampledikk.pkl'
        participants_data_3 = 'noEOG/3-downsampledikk.pkl'
        participants_data_4 = 'noEOG/4-downsampledikk.pkl'
        participants_data_5 = 'noEOG/5-downsampledikk.pkl'
        participants_data_6 = 'noEOG/6-downsampledikk.pkl'
        participants_data_7 = 'noEOG/7-downsampledikk.pkl'
        participants_data_9 = 'noEOG/9-downsampledikk.pkl'
        participants_data_10 = 'noEOG/10-downsampledikk.pkl'
        participants_data_11 = 'noEOG/11-downsampledikk.pkl'
        participants_data_12 = 'noEOG/12-downsampledikk.pkl'
        participants_data_13 = 'noEOG/13-downsampledikk.pkl'

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

        print(len(frontal_1))

        frontal_final = np.concatenate([frontal_1, frontal_2, frontal_3, frontal_4, frontal_5, frontal_6, frontal_7, frontal_9, frontal_10, frontal_11, frontal_12,frontal_13 ])
        lengths = [len(frontal_1),len(frontal_2), len(frontal_3), len(frontal_4), len(frontal_5), len(frontal_6), len(frontal_7),
         len(frontal_9), len(frontal_10), len(frontal_11),len(frontal_12), len(frontal_13)]

        return [frontal_final, lengths]
        #return [frontal_1, len(frontal_1)]

def getTestingTrainingSets(X, lengths):
    """
    The sets yielded are used for cross validation
    :param frontal_final:
    :param lengths:
    :return:
    """

    j = 0
    sum = 0

    for length in lengths:
        fromm = sum
        to = sum + length

        if sum == 0:

            yield [X[fromm:to],X[to:],lengths[j+1:]]

        else:
            lengths_train = []
            test_set = X[fromm:to]
            sub_train_1 = X[0:fromm]
            sub_train_2 = X[to:]

            for i in range(12):

                if j!=i:
                    lengths_train.append( lengths[i] )

            train_set = np.concatenate([sub_train_1, sub_train_2])

            yield [test_set,  train_set,lengths_train]

        sum = sum + length
        j = j +1







def trainModel(X,lengths,states):
    model = GaussianHMM(n_components=states, covariance_type="diag", n_iter=1000).fit(X, lengths)

    print(model.predict(X))
    print(model.monitor_.converged)
    print(model.monitor_)
    score = model.score(X,lengths)

    print(score)
    return [model,score]


def trainModelGMM(X, lengths, states, num_gaus):

    model = GMMHMM(n_components=states, n_mix=num_gaus,n_iter=1000,verbose=True).fit(X,lengths)

    print('Mixture Models + HMM')
    print(model.predict(X))
    print(model.monitor_.converged)
    print(model.monitor_)
    print(model.score(X, lengths))


def saveModel(model,num_state):
     from sklearn.externals import joblib
     joblib.dump(model, "model_"+str(num_state)+"_200.pkl")



datas = loadData(False, "both")
print(len(datas[0]), datas[1])

crossValidation = getTestingTrainingSets(datas[0],datas[1])
for states in range(2, 11):


    print("states number:", states)
    sum_score = 0
    score_model_best = 100000000

    for test, train,train_length in getTestingTrainingSets(datas[0], datas[1]):

        result = trainModel(train,train_length, states)
        model = result[0]
        score_model = result[1]
        if score_model < score_model_best:
            score_model_best = score_model
            saveModel(model, states)

        score = model.score(test)
        print("Score:", model.score(test))
        sum_score +=score

    print("Mean score:",sum_score/12)



"""
from sklearn.externals import joblib
from hmmlearn.hmm import GaussianHMM
import pickle

#model = joblib.load( "model_6_200.pkl")
#events = pickle.load(open("ppn13_events","r"))
#print (events)


#print(model.score(frontal_final,lengths))
predictions = model.predict(datas[0])
A = model.transmat_
means = model.means_
pi = model.startprob_
B = model.predict_proba(datas[0])
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


X = datas[0]
Y = predictions

plt.plot(X[35000:36000])
plt.plot(Y[35000:36000])
plt.show()
print (len(X))
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
"""


import numpy as np
import matplotlib.pyplot as plt





plt.figure(1)
plt.subplot(211)
plt.plot( X[0:300])

plt.subplot(212)
plt.plot( Y[0:300])
plt.show()