# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:17:52 2020

@author: Hafed-eddine BENDIB
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def plot_logvar(logvarData=None, clss=None, offset=0.3, trial_axis=0, colors=[None, None], **kwargs):
    if len(logvarData.keys()) > 2:
        colors = []
        colors += [None] * len(logvarData.keys())
    if logvarData[list(logvarData.keys())[0]].ndim >= 2:
        nCH = logvarData[list(logvarData.keys())[0]].shape[-1]
    else:
        nCH = 1
    if clss is None:
        clss = list(logvarData.keys())
    plt.figure(figsize=(12, 5))
    cls1_comp = np.arange(nCH)
    cls2_comp = np.arange(nCH) + offset
    cls1_mag = np.mean(logvarData[list(logvarData.keys())[0]], axis=trial_axis)
    cls2_mag = np.mean(logvarData[list(logvarData.keys())[1]], axis=trial_axis)

    plt.bar(cls1_comp, cls1_mag, width=0.4, color=colors[0])
    plt.bar(cls2_comp, cls2_mag, width=0.4, color=colors[1])
    plt.xlim(-0.5, nCH + 0.5)
    plt.gca().yaxis.grid(True)
    plt.title(kwargs['title'])
    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.legend(clss)
    plt.show()
    return "plot ok"


def hurst_m1_plot(H=None, c=None, RS=None, window_sizes=None, scale='log', xlabel='Intervalle de temps',
                  ylabel='Taux R/S', label='Droite de régression'):
    fig_, ax_ = plt.subplots()
    ax_.plot(window_sizes, c * window_sizes ** H, color="b", lw=2, label=label)
    ax_.scatter(window_sizes, RS, color="purple", lw=4, label="Données R/S")
    ax_.set_xscale(scale)
    ax_.set_yscale(scale)
    ax_.set_xlabel(xlabel)
    ax_.set_ylabel(ylabel)
    plt.grid()
    plt.legend(loc="best")
    plt.show()


def sampEn_dists_plot(distances, tol, m, title="Sample Entropy", save_fig=None):
    n_std = 3
    n_bins = 50
    data_cnc = np.concatenate(distances)
    mean_cnc = np.mean(data_cnc)
    std_cnc = np.std(data_cnc)
    range_ = (0, mean_cnc + std_cnc * n_std)

    ii = 0
    template = ["red", "green"]
    for h, bin_ in [np.histogram(dist, n_bins, range_) for dist in distances]:
        width_ = bin_[1] - bin_[0]
        plt.bar(bin_[:-1], h, width_, label="m:{:d}".format(m + ii), color=template[ii], alpha=0.8)
        ii += 1
    plt.axvline(tol, color="blue", lw=3.5)

    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.show()
    if save_fig is not None:
        plt.savefig(save_fig)


def features_scatter_plot(left, right, clss_=['left', 'right'], colors=['r', 'g'], boundary_decision_plot=False,
                          W=None, b=None, title='Données d\'entraînement', offset_xy=[0.5, 0.5]):
    fig, axs = plt.subplots(1, 1)
    axs.scatter(left[:, 0], left[:, -1], color=colors[0])
    axs.scatter(right[:, 0], right[:, -1], color=colors[1])
    axs.set_xlabel('1 ère Composante', fontsize=14)
    axs.set_ylabel('Dernière Composante', fontsize=14)
    plt.title(title)
    axs.legend(clss_)
    if boundary_decision_plot:
        xlim_min = min(left[:, -1].min(), right[:, -1].min()) - offset_xy[0]
        xlim_max = max(left[:, -1].max(), right[:, -1].max()) + offset_xy[0]
        ylim_min = min(left[:, 0].min(), right[:, 0].min()) - offset_xy[1]
        ylim_max = max(left[:, 0].max(), right[:, 0].max()) + offset_xy[1]
        # Calculate decision boundary (x,y)
        x = np.arange(xlim_min, xlim_max, 0.1)
        y = (b - W[0] * x) / W[1]

        # Plot the decision boundary
        plt.plot(x, y, linestyle='--', linewidth=2, color='b')
        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
    plt.show()


def classes_plot(inputData=None, labels=None, components=[0, -1], colors=None,
                 boundary_plot=False, W=None, B=None, *args, **kwargs):
    first, last = components
    clss = list(inputData.keys())
    min_ = [np.inf] * len(clss) * 2
    max_ = [np.inf] * len(clss) * 2
    if labels is None:
        labels = ["left", "right"]
    fig = plt.figure(**kwargs)
    fig.suptitle("Features Distribution")
    ax = fig.add_subplot(111)
    for idx, cls_ in enumerate(clss):
        ax.scatter(inputData[cls_][:, first], inputData[cls_][:, last],
                   edgecolors="k", c=colors[idx], s=100, label=cls_)

        min_[idx] = min(min_[idx], inputData[cls_][:, first].min())
        min_[idx + len(clss)] = min(min_[idx + len(clss)], inputData[cls_][:, last].min())
        max_[idx] = min(max_[idx], inputData[cls_][:, first].max())
        max_[idx + len(clss)] = min(max_[idx + len(clss)], inputData[cls_][:, last].max())

    ax.set_xlabel("First Component")
    ax.set_ylabel("Last Component")
    ax.legend(labels, loc="best")
    ax.grid(True)
    if boundary_plot:
        x = np.arange(min(min_[:len(clss)]), max(max_[:len(clss)]), 0.1)
        y = (B - W[0] * x) / W[1]
        ax.plot(x, y, linestyle="--", lw=2, color="k")
        ax.axis([min(min_[:len(clss)]) - 0.1, max(max_[:len(clss)]) + 0.1, min(min_[len(clss):]) - 0.1,
                 max(max_[len(clss):]) + 0.1])
    plt.show()
    return "plot ok"


def pca_selector_plot(split_data=None, fig=None, axis=None, title=None, labels=None, cmap=None, *args, **kwargs):
    if cmap is None:
        cmap = matplotlib.colors.ListedColormap(["red", "green"])
    if fig is None:
        fig = plt.figure(figsize=(6, 5))
    fig.suptitle(title)
    for idx, pos in enumerate(range(221, 225)):
        if pos == 221:
            projection = "3d"
            axis.append(fig.add_subplot(pos, projection=projection))
            axis[idx].scatter(xs=split_data[0][:, 0], ys=split_data[0][:, 1],
                              zs=split_data[0][:, 2], c=split_data[2],
                              cmap=cmap, depthshade=True, *args, **kwargs)
            axis[idx].set_xlabel(labels[0])
            axis[idx].set_ylabel(labels[1])
            axis[idx].set_zlabel(labels[2])
            axis[idx].grid(True)
            plt.tight_layout()
        else:
            if idx == 3:
                idx1 = idx - 1
                idx2 = 0
            else:
                idx1 = idx - 1
                idx2 = idx
            projection = None
            axis.append(fig.add_subplot(pos))
            axis[idx].scatter(x=split_data[0][:, idx1],
                              y=split_data[0][:, idx2], c=split_data[2],
                              cmap=cmap, *args, **kwargs)
            axis[idx].set_xlabel(labels[idx1])
            axis[idx].set_ylabel(labels[idx2])
            axis[idx].grid(True)
            plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print('This is a test section for features plotter module')
