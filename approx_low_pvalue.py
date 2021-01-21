#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:25:47 2019

@author: rasmus
"""


__author__ = 'Rasmus Magnusson'
__COPYRIGHT__ = 'Rasmus Magnusson, 2018, Linköping'


import numpy as np


def get_p(z):
    """ This function approximates low p values of the normal distribution. 
    For a Z-value, the 10-logarithm of the p-value is returned. """
    return (.5*np.log(2*np.pi) - 0.5*z*z - np.log(z) + np.log(1 - z**-2 + 3*z**-4))/(np.log(10))
    
    
def CLT_binom(vals, p=0.5):
    """ This function can be used to get low p-values where the precision of 
    the binomial tests fails. It relies on the central limit theorem, so be
    sure to only use for large sample sizes """
    n = len(vals)
    mu = n*p
    sigma = np.sqrt(mu*(1-p))
    Z = (vals.sum() - mu)/sigma
    return get_p(Z)    