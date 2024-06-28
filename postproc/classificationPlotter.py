# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 19:56:42 2020

@author: Hafed-eddine BENDIB
"""
import matplotlib.pyplot as plt


def knn_k_values_plot(scores, x_ax, title='', xlabel='', ylabel=''):
    plt.figure()
    plt.plot(x_ax, scores, lw=1.5, color='blue', marker="o")
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.show()


def cv_parameter_plot(k_scores, x_ax, title='', xlabel='', ylabel=''):
    plt.figure()
    plt.plot(x_ax, k_scores, lw=1.5, color='purple', marker="o")
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.show()
