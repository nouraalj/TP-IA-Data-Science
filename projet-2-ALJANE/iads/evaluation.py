# -*- coding: utf-8 -*-

"""
Package: iads
File: evaluation.py
Année: LU3IN026 - semestre 2 - 2022-2023, Sorbonne Université
"""

# ---------------------------
# Fonctions d'évaluation de classifieurs

# import externe
import numpy as np
import pandas as pd
import copy

# ------------------------ 
def crossval(X, Y, n_iterations, iteration):
    start, end = iteration*int(len(Y)/n_iterations), (iteration+1)*int(len(Y)/n_iterations)
    Xtrain = np.delete(X, np.s_[start:end], axis=0)
    Ytrain = np.delete(Y, np.s_[start:end], axis=0)
    Xtest = X[start:end]
    Ytest = Y[start:end]
    return Xtrain, Ytrain, Xtest, Ytest


def crossval_strat(X, Y, n, i):
    Xtrains, Ytrains, Xtests, Ytests = [], [], [], []
    for y in np.unique(Y):
        Xtrainy, Ytrainy, Xtesty, Ytesty = crossval(X[Y==y], Y[Y==y], n, i)
        Xtrains.append(Xtrainy)
        Ytrains.append(Ytrainy)
        Xtests.append(Xtesty)
        Ytests.append(Ytesty)
    return (np.concatenate(Xtrains), np.concatenate(Ytrains),
            np.concatenate(Xtests), np.concatenate(Ytests))


def analyse_perfs(L):
    """ L : liste de nombres réels non vide
        rend le tuple (moyenne, écart-type)
    """
    return np.mean(L), np.std(L)   

def validation_croisee(C, DS, nb_iter):
    """ Classifieur * tuple[array, array] * int -> tuple[ list[float], float, float]
    """
    X, Y = DS   
    perf = []
    
    
    
    ########################## COMPLETER ICI 
    for i in range(nb_iter):
        Xapp,Yapp,Xtest,Ytest = crossval_strat(X, Y, nb_iter, i)
        newC = copy.deepcopy(C)
        newC.train(Xapp, Yapp)
        perf.append(newC.accuracy(Xtest, Ytest))

    
    ##########################
    (perf_moy, perf_sd) = analyse_perfs(perf)
    return (perf, perf_moy, perf_sd)
