#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 08:38:17 2019

@author: rasmus
"""


__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Rasmus Magnusson, 2019, Linköping'



import pandas as pd
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
import h5py
from approx_low_pvalue import get_p
import os



def read_adj(path):
    adj = pd.read_csv(path)
    adj = adj.set_index(adj.columns[0])
    adj.values[np.isinf(adj)] = np.nan
    return adj.transpose()


def read_GS():
    GS = pd.read_csv('trrust_rawdata.human.tsv', sep='\t', header=None).iloc[:,:3]
    GS = GS[(GS.iloc[:,2] != 'Unknown')]
    return GS


def get_enr(adj, GS):
    unique_targets = GS.iloc[:,1].unique()
    GS = GS.set_index(GS.columns[1])
    vals = []
    for tname in unique_targets:
        regTFs = GS.loc[tname].transpose().values[0]
        tmp_vals = adj.loc[tname].loc[regTFs]
        if type(tmp_vals) == pd.Series:
            for v in tmp_vals:
                vals.append(v)
        else:
            vals.append(tmp_vals)
    return np.array(vals)


def sortnames(names):
    n_hidden = []
    n_layers = []
    for n in names:
        if 'no_hidden' in n:
            n_hidden.append(0)
            n_layers.append(0)
            continue
        n = n[:-4].replace('TFs_to_targets_', '').split('_')
        n_hidden.append(int(n[0]))
        n_layers.append(len(n))
    return np.array(n_layers), np.array(n_hidden)

def plot_df(df):
    n_layers, n_hidden = sortnames(df.columns.copy())
    df.iloc[1,:] = -np.log10(df.iloc[1,:])
    _golden = (1 + (5**(.5)))/2
    f, ax = plt.subplots(1,1,figsize=(_golden*4, 4))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_position(('outward', 20))
    place = 0

    name_pos = []
    name = []
    for lvl in np.unique(n_layers):
        tmp_mean_vals = df.iloc[1, n_layers == lvl]
        nParams_tmp = n_hidden[n_layers == lvl]
        names_tmp = n_hidden[n_layers == lvl]
        for i in np.argsort(nParams_tmp):#range(tmp_mean_vals.shape[0]):
            ax.bar(place, tmp_mean_vals[i], color='#850000ff', label='old median')
            name_pos.append(place)
            name.append(names_tmp[i])
            place += 1.2
        place = place + 2
    ax.set_xticks(name_pos)
    name[0] = 'No nodes'
    ax.set_xticklabels(name, rotation=90)
    ax.tick_params(labelsize=17)
    ax.tick_params(width=2, length=4)
    ax.tick_params(width=2, length=4, axis='y', labelsize=17)
    ax.set_ylabel(r'-log$_{10}$ p', fontsize=18)
    f.savefig('img/correct_sign_p.png')

def main():
    files = os.listdir()
    files = [fname for fname in files if 'TFs' in fname]

    enr = {}
    all_res_p = {}
    OR = {}
    acc = {}
    for fname in files:
        adj = read_adj(path=fname)

        if adj.shape[0] < adj.shape[1]:
            adj = adj.transpose()
        GS = read_GS()


        GS = GS[GS.iloc[:,1].isin(adj.index)]
        GS = GS[GS.iloc[:,0].isin(adj.columns)]

        adj = adj[adj.index.isin(GS.iloc[:,1].values)]
        adj = adj.iloc[:, adj.columns.isin(GS.iloc[:,0].values)]

        GS_pos = GS[(GS.iloc[:,2] == 'Activation')]
        GS_neg = GS[(GS.iloc[:,2] == 'Repression')]

        posvals = get_enr(adj.copy(), GS_pos)
        negvals = get_enr(adj.copy(), GS_neg)

        (posvals > 0).sum()/len(posvals)
        (negvals < 0).sum()/len(negvals)

        corrdir = ((posvals > 0).sum() + (negvals < 0).sum())
        tot = len(negvals) + len(posvals)
        p = sts.binom_test(corrdir, tot, alternative='greater')

        all_res_p[fname] = p
        enr[fname] = corrdir/(0.5*tot)
        OR[fname] = corrdir/(tot - corrdir)
        acc[fname] = corrdir/tot
    df = pd.DataFrame([enr, all_res_p, OR, acc])
    df.to_csv('direction_TRRUST_res.csv')

    plot_df()
main()
