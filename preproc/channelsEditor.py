# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np


def reref_channels(inputData=None, axes=None, ref_chan="ch1", ch_ax=-1, t_ax=0):
    data = inputData.copy()
    chan_list = []
    for ch in axes[ch_ax]:
        chan_list.append(ch.lower())  # ignore the upper/lower case inputs.
    pos = chan_list.index(ref_chan.lower())  # convert it to lower.
    ref_chan_data = (data[:, pos]).reshape(data.shape[t_ax], 1)
    data = data - ref_chan_data
    return data


def signal_concat(*args, **kwargs):
    concat_sig = np.concatenate(*args, **kwargs)
    return concat_sig


if __name__ == "__main__":
    print(f"This is the test section of channels editor module")
