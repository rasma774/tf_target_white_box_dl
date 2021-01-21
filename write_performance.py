#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:09:42 2020

@author: rasmus
"""


from keras.models import load_model
import numpy as np
import pandas as pd
import os

__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Rasmus Magnusson, 2020, Linköping'

def calc_R2(pred, data):
    SSres = np.sum((pred - data)**2, 0)
    SStot = np.sum((data - data.mean())**2, 0)
    return 1 - (SSres/SStot)


def main():
    model_names = [x for x in os.listdir() if '.h5' in x]

    R2 = {}
    R2DF = pd.DataFrame()
    for MNAME in model_names:
        R2[MNAME] = []
    for i in [90]:
        FNAME = str(i)
        TFs = pd.read_csv('../val_data/tf_' + FNAME + '.csv').iloc[:,1:].values[:600,:]
        targetd = pd.read_csv('../val_data/target_' + FNAME + '.csv').iloc[:,1:].values[:600,:]
        for MNAME in model_names:
            print(MNAME)
            model = load_model(MNAME)
            pred = model.predict(TFs)
            r2tmp = calc_R2(pred, targetd)
            R2[MNAME].append(r2tmp)
            R2DF[MNAME] = pd.Series(r2tmp)
        R2DF.to_csv('R2/R2_targets_dataset_' + str(i) + '.csv')

if __name__ == '__main__':
    main()
