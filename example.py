# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:22:17 2023

@author: -
"""
import os
import pickle
import numpy as np
import time as pytime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.ticker as ticker

from src.plotting import create_polar_plot, add_cbar
from unet_vda import Unet_VDA
vda = Unet_VDA()


def angle_diff(angle1, angle2=None):
    diff = np.diff(angle1) if angle2 is None else angle2-angle1
    return (diff+180) % 360 - 180


#%%
filenames = os.listdir('data')
# 1 represents velocity scan with 2 data gaps
# 2 represents scan with use of 2 different Nyquist velocities
# 3 represents scan that spans much more than 360 degrees
# 4 represents scan with half of it missing
with open('data/'+filenames[1], 'rb') as f:
    data, azis, vn = pickle.load(f).values()

diffs = -angle_diff(azis[::-1])
csum = np.cumsum(diffs)
# Ensure that azis spans less than 360°, to prevent issues with unsampled radials in self.map_onto_uniform_grid
n_azi = len(azis) if csum[-1] < 360 else 1+np.where(csum >= 360)[0][0]
data, azis, vn = data[-n_azi:], azis[-n_azi:], vn[-n_azi:]

diffs = diffs[:n_azi-1][::-1]
da_median = np.median(diffs)
da = diffs[diffs < 3*da_median].mean()

t = pytime.time()
data_new = vda(data, vn, azis, da)
print(pytime.time()-t, '')


data = vda.expand_data_to_360deg(data, vn, azis, da)[0]
data_new = vda.expand_data_to_360deg(data_new, vn, azis, da)[0]

# Set Colormap
cmap=cm.get_cmap('seismic')
cmap.set_under([.9,.9,.9])
cmap.set_bad([.9,.9,.9])
cmap.set_over([.9,.9,.9])
norm=mpl.colors.Normalize(vmin=-70, vmax=70)  

a=1
fig,axs=plt.subplots(1,2,figsize=(15*a,5*a))
im,_ = create_polar_plot(axs[0], data,cmap,norm,0.5,annotate=False)
im,_ = create_polar_plot(axs[1], data_new,cmap,norm,0.5,annotate=False)
add_cbar(fig,axs[1],im,'m/s')

for k,ax in enumerate(axs):
        ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
        ax.xaxis.set_major_formatter(ticks_x)
        ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/1000))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.set_xlabel('km')
axs[0].set_ylabel('km')
axs[0].set_title('Level 2 Velocity (Aliased)')
axs[1].set_title('U-Net Result')

plt.tight_layout()