# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 16:22:17 2023

@author: -
"""
import os
import pickle
import numpy as np

from unet_vda import Unet_VDA
vda = Unet_VDA()


def angle_diff(angle1, angle2=None):
    diff = np.diff(angle1) if angle2 is None else angle2-angle1
    return (diff+180) % 360 - 180


#%%
filenames = os.listdir('data')
print(filenames)
with open('data/'+filenames[0], 'rb') as f:
    data, azis, vn = pickle.load(f).values()

diffs = -angle_diff(azis[::-1])
csum = np.cumsum(diffs)
# Ensure that azis spans less than 360°, to prevent issues with unsampled radials in self.map_onto_uniform_grid
n_azi = len(azis) if csum[-1] < 360 else 1+np.where(csum >= 360)[0][0]
data, azis, vn = data[-n_azi:], azis[-n_azi:], vn[-n_azi:]

diffs = diffs[:n_azi-1][::-1]
da_median = np.median(diffs)
da = diffs[diffs < 3*da_median].mean()

data_new = vda(data, vn, azis, da)