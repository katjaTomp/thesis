import mne
import matplotlib
import numpy as np

import pandas

#### Set up matplotlib ####
matplotlib.use('TkAgg')
matplotlib.interactive(False)


### fix the 50 ms delay by adding the 50ms to the events first column values
## mne converts seconds to secs * sample freq therefore, I will add 0.05 sec * 2048 Hz = 102,4 ~ 102 to each timestamp in the events

def add50ms(events):

    for i in range(len(events)):
        events[i][:1:1] += 102

    return events


#### Load data ####

def loadFile(file):
    raw = mne.io.read_raw_edf(file, eog=['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6'], stim_channel=-1, preload=True)  # documentation here:https://martinos.org/mne/stable/generated/mne.io.read_raw_edf.html#mne.io.read_raw_edf


    findEvents(raw)

    #print(raw.plot(block=True))

    return raw


### Bad Channels are included here ####

def addBadChannels(raw):
    raw.info['bads'] = ['EXG7', 'EXG8']
    #print(raw.plot(block=True))

    return raw


### Events ####

def findEvents(raw):
    events = mne.find_events(raw, shortest_event=1 , output="onset",consecutive='increasing',  stim_channel="STI 014")#,

    print(events)
    events = add50ms(events)
    print(events)
    return events


### Find picks ###

def findPicks(raw):
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
    return picks



###filters applied

def highPassfilter(raw, picks, l_freq):
    raw = raw.filter(l_freq=l_freq, h_freq=None, picks=picks)
    return raw


def lowPassfilter(raw, picks, h_freq):
    raw = raw.filter(h_freq=h_freq, l_freq=None, picks=picks)
    return raw


## average reference ###
# Data were referenced to the average of left and right mastoid signal.

def averRef(raw):
    raw = raw.set_eeg_reference(ref_channels='average', projection=False)

    return raw


def eogRef(raw):
    raw = raw.set_eeg_reference(ref_channels=['EXG5', 'EXG6'], projection=False)
    return raw


#### Epoching ####

def genEpochs(raw, events, picks):
    event_id, tmin, tmax =  None, 0, 2.34  # was=0 0.5
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0),
                        # ,reject=dict( eog=150e-6,eeg=80e-6)
                        preload=True)
    return epochs


#### Retrieve only particular channel ###

def getChannel(channel, raw):
    pick_chans = [channel]

    specific_chans = raw.copy().pick_channels(pick_chans)

    print(specific_chans.plot(block=True))

    return specific_chans


### Artifact reduction SSP/PCA projections of ECG artifacts ###

def applyNotchFilter(raw):
    raw.notch_filter(freqs=50)
    return raw


def rejectEOG(raw):
    eog_events = mne.preprocessing.find_eog_events(raw)
    n_blinks = len(eog_events)
    # Center to cover the whole blink with full duration of 0.5s:
    onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)
    raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                      orig_time=raw.info['meas_date'])
    print(raw.annotations)  # to get information about what annotations we have
    raw.plot(events=eog_events, block=True)


def rejectBadEpochs(raw):
    reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)
    events = mne.find_events(raw, stim_channel='STI 014', shortest_event=0)
    event_id = [65281 ,65284 ,65286 ,65288 ,65294 ,65296 ,65298 ,65304 ,65306 ,65308]
    tmin = -0.2  # start of each epoch (200ms before the trigger)

    tmax = 0.5  # end of each epoch (500ms after the trigger)
    baseline = (None, 0)  # means from the first instant to t = 0
    picks_meg = mne.pick_types(raw.info, meg=True, eeg=True, eog=True,
                               stim=False, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks_meg, baseline=baseline, reject=reject,
                        reject_by_annotation=True)
    epochs.drop_bad()
    print(epochs.drop_log[40:45])  # only a subset
    epochs.plot_drop_log()

    return epochs


def baseline(epochs, time):
    epochs.apply_baseline(baseline=(None, time), verbose=None)

    return epochs


## reducing the sampling frequency


def downsampleEpochs(freq, epochs):
    epochs_resampled = epochs.copy().resample(freq, npad='auto')

    return epochs_resampled


# Data pre-processing
filenames = ['ppn12_23mei.bdf','PPN13_24MEI.bdf']
#filenames = ['PPN2_10mei.bdf','PPN3_12mei.bdf','ppn4_18mei.bdf','ppn5_18mei.bdf','PPN6_19mei.bdf','PPN7_19mei.bdf','PPN9-20mei.bdf','PPN10-20mei.bdf','PPN11_20mei.bdf','PPN2_10mei.bdf','PPN3_12mei.bdf','ppn4_18mei.bdf','ppn5_18mei.bdf','PPN6_19mei.bdf','PPN7_19mei.bdf']
i = 12
for filename in filenames:
    raw = loadFile(filename)

    raw = addBadChannels(raw)  # channels with no signal (to check for all participants)

    raw = applyNotchFilter(raw)  # 50 Hz notch filter
    picks = findPicks(raw)
    raw = highPassfilter(raw, picks, 0.16)  # High pass filter 0.16 Hz
    raw = lowPassfilter(raw, picks, 30)  # Low pass filter 30 Hz

###takes too time see how to save it somewhere for fast retrieval

    raw = eogRef(raw)  # reference data to two mastoids

#print(raw.plot(block=True))

#rejectEOG(raw)

#### Epoching ####
    events = findEvents(raw)
    epochs = genEpochs(raw, events, picks)

    epochs = baseline(epochs,
                  0.1)  # The data were baseline corrected over the 100 ms interval preceding the stimulus presentation.

   # print(epochs.plot(picks=picks, block=True))
    print("downsampling epochs")

    epochs = downsampleEpochs(200, epochs)




    data = epochs.get_data()

# save the data as dataframe

#mne.viz.plot_events(events=events)
    print(data)

# putting epochs into data frame

    index = ['epoch','time']  # 1, dict(grad=1e13)

    df = epochs.to_data_frame(picks=None, scalings=None, index=index)
    df.to_pickle(str(i)+'-downsampledikk.pkl')
    i+=1

    print(df.dtypes)


