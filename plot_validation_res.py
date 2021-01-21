#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:33:54 2019

@author: rasmus
"""


__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Rasmus Magnusson, 2019, Linköping'


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def read_and_plot_results(fname = 'results_TRRUST.txt'):
    f = open(fname, 'r')
    text = f.read().split('\n')
    f.close()
    res = {}

    last = ''
    for line in text:
        if last == '' and line != '':
            current_name = line[:-4]
            res[current_name] = []
        elif line != '':
            res[current_name].append(float(line.split(' ')[-1]))
        last = line
    return pd.DataFrame(res)


def main():
    TRRUST = read_and_plot_results('results_TRRUST.txt')
    ERNST = read_and_plot_results('results.txt')

    labels = TRRUST.columns
    labels = [modname.split('targets_')[-1] for modname in labels]
    labels = np.array(labels)
    labels[labels == 'no_hidden_'] = '0'

    nParam_tmp = [np.array(name.split('_')).astype(int).prod() for name in labels]
    pos = np.argsort(nParam_tmp)
    labels = np.array(labels)[pos]
    TRRUST = TRRUST.iloc[:, pos]
    ERNST = ERNST.iloc[:, pos]

    _golden = (1+5**.5)/2
    f, ax = plt.subplots(1,1,figsize=(_golden*4, 4))
    ax.tick_params(labelsize=17)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_position(('outward', 20))
    ax.bar(np.arange(TRRUST.shape[1]) - .33/2, TRRUST.values[-2,:], width=.33, color='#85002bda', label='TRRUST')
    ax.bar(np.arange(ERNST.shape[1]) + .33/2, ERNST.values[-2,:], width=.33, color='#008568ff', label='ERNST')
    ax.legend()
    ax.set_xticklabels(labels, rotation=90)
    ax.set_xticks(np.arange(ERNST.shape[1]))
    ax.set_ylabel(r'$\sigma$ enrichment', fontsize=17)
    f.savefig('validation_enrich.svg', bbox_inches='tight')
    f.savefig('validation_enrich.png', bbox_inches='tight')

    ax.plot([-.5, np.arange(ERNST.shape[1])[-1] + .5], [1, 1], '--', color='k')
    f.savefig('validation_enrich_v2.svg', bbox_inches='tight')
    f.savefig('validation_enrich_v2.png', bbox_inches='tight')
main()
