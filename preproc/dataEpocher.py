# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 10:31:17 2019

Author: Hafed-eddine BENDIB
"""

import numpy as np


def data_epoching(inputData=None, axes=None, input_markers=None, markers_dict=None, epoch_start=0, epoch_end=None,
                  samples_n=None, sfreq=256, t_ax=0):
    if epoch_start > epoch_end:
        raise ValueError("Vérifier les bornes de l'époque !")
    if input_markers is None:
        raise ValueError("Les données doivent contenir des marqueurs !")
    if samples_n is not None:
        if samples_n < 0:
            raise ValueError("la paramètre 'samples_n' doit être positif !")
        samples_n_t = axes[t_ax][-samples_n:] if samples_n > 0 else []
    N_samples = int(sfreq * (epoch_end - epoch_start) / 1000)
    epoched_data = []
    epochs_classes = []
    epoch_class_dict = {}
    classNames = sorted(markers_dict.keys())
    indices = []
    for marker, t_init in input_markers:
        for class_order, c_name in enumerate(classNames):
            if marker in markers_dict[c_name]:
                indice_class = (axes[t_ax] >= t_init + epoch_start) & (axes[t_ax] < t_init + epoch_end)
                indice_class = np.where(indice_class == True)[0]
                if len(indice_class) != N_samples:
                    continue
                times = axes[t_ax].take(indice_class)
                if samples_n is not None:
                    if samples_n == 0:
                        continue
                    if len(np.intersect1d(times, samples_n)) == 0 and (t_init < samples_n_t[0]):
                        continue
                indices.append(indice_class)
                epochs_classes.append(class_order)
                epoch_class_dict.update({class_order: c_name})

    if len(indices) == 0:
        epoched_data = np.array([])
    else:
        epoched_data = inputData.take(indices, axis=t_ax)
        epoched_data = epoched_data.swapaxes(0, t_ax)
    epoched_time = np.linspace(epoch_start, epoch_end, N_samples)
    return epoched_data, epoched_time, epochs_classes, epoch_class_dict


def signal_clipping(inputData=None, clip_sense="start", split_data=True, sfreq=256, t_ax=0):
    if clip_sense == "start":
        [idx, step] = [0, 1]
    else:
        [idx, step] = [-1, -1]
    segment_clip = (np.size(inputData, axis=t_ax)) % sfreq
    clipped_signal = np.delete(inputData, slice(idx, idx + (step * segment_clip), step), axis=t_ax)
    if split_data:
        splitted_signal = np.array(np.split(clipped_signal,
                                            np.size(clipped_signal, axis=t_ax) // sfreq))
        return splitted_signal.T, clipped_signal
    else:
        return clipped_signal


if __name__ == "__main__":
    print(f"This is the test section of epocher module")
