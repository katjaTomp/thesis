### Plotting
import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from matplotlib import cm, pyplot as plt
import numpy as np
cmap = cm.hot

colours = cm.rainbow(np.linspace(0,1,3))
values = [1,3,4,5,5,5,5,5,4,4,4,3,1]
states = [1,1,1,2,2,2,2,2,1,1,1,1,1]

plt.plot( values, states)
plt.show()
