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
    adj = np.abs(adj)
    return adj.transpose()


def read_GS():
    GS = pd.read_csv('trrust_rawdata.human.tsv', sep='\t', header=None).iloc[:,:2]
    return GS

def main():
    files = os.listdir()
    files = [fname for fname in files if 'TFs' in fname]

    all_res = {}
    all_res_p = {}
    for fname in files:
        adj = read_adj(path=fname)

        if adj.shape[0] < adj.shape[1]:
            adj = adj.transpose()
        GS = read_GS()


        GS = GS[GS.iloc[:,1].isin(adj.index)]
        GS = GS[GS.iloc[:,0].isin(adj.columns)]

        adj = adj[adj.index.isin(GS.iloc[:,1].values)]
        adj = adj.iloc[:, adj.columns.isin(GS.iloc[:,0].values)]


        unique_targets = GS.iloc[:,1].unique()
        unique_tfs = GS.iloc[:,0].unique()
        GS = GS.set_index(GS.columns[1])
        vals = []
        p = []
        for tname in unique_targets:
            regTFs = GS.loc[tname].transpose().values[0]
            not_reg = unique_tfs[~np.in1d(unique_tfs, regTFs)]
            tmp_vals = adj.loc[tname].loc[regTFs]
            tmp_vals_not = adj.loc[tname].loc[not_reg]

            vals.append(np.median(tmp_vals)/np.median(tmp_vals_not))

            if not type(tmp_vals) == pd.Series:
                tmp_vals = [tmp_vals]

            # catch when all are same
            try:
                p.append(sts.mannwhitneyu(tmp_vals, tmp_vals_not.values, alternative='greater')[1])
            except:
                p.append(np.nan)

        p = np.array(p)
        all_res_p[fname] = p

        vals = np.array(vals)
        vals[np.isnan(vals)] = 1
        print(np.nanmedian(vals))
        all_res[fname] = vals

        with open('results_TRRUST_individual.txt', 'a') as f:
            f.write(fname)
            f.write('\nMedian enrichment: ' + str(np.nanmedian(vals)))
            f.write('\n\n')
    pd.DataFrame(all_res).to_csv('trrust_individual_lightup_results.csv')
    pd.DataFrame(all_res_p).to_csv('trrust_individual_lightup_results_prob.csv')
main()
