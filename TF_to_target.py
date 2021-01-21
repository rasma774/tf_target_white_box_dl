#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 08:22:16 2019

@author: rasmus
"""


__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Rasmus Magnusson, 2019, Linköping'


import pandas as pd
import numpy as np
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras import losses
import argparse
import os

from plot_functions import plot_r2_dist, plot_r2_dist_cumulative

def parse_them_arguments():
    parser = argparse.ArgumentParser(
            description='This is an example script of how to use the argparser in Python'
            )
    parser.add_argument('--Nlayers', default=[2], type=int, nargs=1, help='Number of layers in the model')
    parser.add_argument('--Nnodes', type=int, nargs='+', help='Number of nodes in the model')

    # We want a parser that reads a textfile of integers and adds one (+1)
    # We ignore all

    args = parser.parse_args()

    Nlayers = args.Nlayers[0]
    Nnodes = args.Nnodes

    if len(Nnodes) != 1:
        assert len(Nnodes) == Nlayers
    else:
        Nnodes = [Nnodes[0] for i in range(Nlayers)]

    return Nnodes


def calcR2(predTarget, pdataTarget):
    SSres = ((predTarget - pdataTarget)**2).sum(0)
    SStot = ((predTarget - pdataTarget.mean(0))**2).sum(0)
    return 1 - (SSres/SStot)


def main():
    Nnodes = parse_them_arguments()
    info = ''
    for i in Nnodes:
        info += '_' + str(i)

    FNAME = '0'
    TFs = pd.read_csv('../data/tf_' + FNAME + '.csv').iloc[:,1:].values
    targetd = pd.read_csv('../data/target_' + FNAME + '.csv').iloc[:,1:].values

    try:
        model = load_model('TFs_to_targets' + info + '.h5')
        print('loaded model')
    except:
        # Creating the model
        model = Sequential()
        model.add(Dense(units=Nnodes[0], activation='elu', input_dim=TFs.shape[1]))

        if len(Nnodes) > 1:
            for i in range(len(Nnodes) - 1):
                model.add(Dense(units=Nnodes[i + 1], activation='elu'))

        model.add(Dense(units=targetd.shape[1], activation='elu'))

        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,decay=0.01)
        model.compile(loss=losses.mean_squared_error,
                          optimizer=adam,
                          metrics=['accuracy'], )

    name_prefix = 'first'


    for i in range(1000):
        Ndatas = int(len(os.listdir('../data/'))/2)
        FNAME = str(np.random.choice(Ndatas,1)[0])
        model.fit(TFs, targetd, epochs=100, batch_size=50, validation_split=0.1)
        model.save('TFs_to_targets' + info + '.h5')

        pred = model.predict(TFs)

    # =============================================================================
    #     SSres = np.sum((pred - TFs)**2)
    #     SStot = np.sum((TFs - TFs.mean())**2)
    #     r2 = 1 - (SSres/SStot)
    #     print(r2)
    # =============================================================================
        r2 = calcR2(pred, targetd)
        #plt.hist(r2[~np.isnan(r2)])

        #plt.hist(r2[r2>-2], bins=40)
        plot_r2_dist(r2, 'img/R2_target_to_TFs_' + name_prefix + info + '.svg')
        plot_r2_dist_cumulative(r2[~np.isnan(r2)], 'img/R2_target_to_TFs_cumulative_' + name_prefix + info + '.svg')

        name_prefix = 'last'
        TFs = pd.read_csv('../data/tf_' + FNAME + '.csv').iloc[:,1:].values
        targetd = pd.read_csv('../data/target_' + FNAME + '.csv').iloc[:,1:].values
        with open('TFs_to_targets' + info + '_learning.txt', 'a') as learn_file:
            for learn_pos in range(len(model.history.history['acc'])):
                learn_file.write(str(model.history.history['loss'][learn_pos]))
                learn_file.write(',')
                learn_file.write(str(model.history.history['acc'][learn_pos]))
                learn_file.write(',')
                learn_file.write(str(model.history.history['val_loss'][learn_pos]))
                learn_file.write(',')
                learn_file.write(str(model.history.history['val_acc'][learn_pos]))
                learn_file.write('\n')

if __name__ == '__main__':
    np.random.seed(0)
    main()
