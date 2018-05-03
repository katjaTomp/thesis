import pandas as pd
import numpy as np

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

files = ['1-downsampledikk.pkl',
         '2-downsampledikk.pkl',
         '3-downsampledikk.pkl',
         '4-downsampledikk.pkl',
         '5-downsampledikk.pkl',
         '6-downsampledikk.pkl',
         '7-downsampledikk.pkl',
         '9-downsampledikk.pkl',
         '10-downsampledikk.pkl',
         '11-downsampledikk.pkl',
         '12-downsampledikk.pkl',
         '13-downsampledikk.pkl']

participants_data_1 = '13-downsampledikk.pkl'
ppn1 = pd.read_pickle(participants_data_1)
ppn1 = ppn1[['condition']]



ppn1_array = np.array(ppn1)

import pickle
pickle.dump(ppn1_array, open('ppn13_events','w'))














