#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 10:42:41 2019

@author: rasmus
"""

from keras.models import load_model
import numpy as np
import pandas as pd
import os


__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Rasmus Magnusson, 2019, Linköping'

def remove_faults(matrix):
    matrix[np.isnan(matrix)] = 0
    matrix[np.isinf(matrix)] = 0
    return matrix

def write_sigma(model, model_name):
    print(model_name)
    inp = np.zeros((model.input_shape[1],model.input_shape[1])).T

    target_names = pd.read_csv('gnames.txt', header=None).values.T[0]
    TFnames = pd.read_csv('tfnames.txt', header=None).values.T[0]

    tf_data = pd.read_csv('../../data/tf_0.csv').iloc[:,1:].mean(0)
    for i in range(inp.shape[1]):
        inp[i, :] = tf_data.values

    half = inp.copy()
    double = inp.copy()
    half[np.eye(half.shape[0]).astype(bool)] = 0.5*half[np.eye(half.shape[0]).astype(bool)]
    double[np.eye(double.shape[0]).astype(bool)] = 2*double[np.eye(double.shape[0]).astype(bool)]

    background_pred = model.predict(inp)
    double_pred = model.predict(double)
    half_pred = model.predict(half)

    double_normed = np.log(double_pred/background_pred)
    half_normed = np.log(half_pred/background_pred)

    #double_normed = remove_faults(double_normed)
    #half_normed = remove_faults(half_normed)

    output = double_normed - half_normed
    output = pd.DataFrame(double_normed)
    output.index = TFnames
    output.columns = target_names
    output.to_csv(model_name + '.csv')



def main():
    files = os.listdir('../')
    files = [fname[:-3] for fname in files if '.h5' in fname]
    for model_name in files:

        model = load_model('../' + model_name + '.h5')
        write_sigma(model, model_name)

if __name__ == '__main__':
    main()
