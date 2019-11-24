#########################################################
## Stat 202A - Homework 9
## Author: Stephanie Stacy
## Date : 12/6/17
## Description: This script implements a support vector machine, an adaboost classifier
#########################################################

import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split


def prepare_data(valid_digits=np.array((6, 5))):
    ## valid_digits is a vector containing the digits
    ## we wish to classify.
    ## Do not change anything inside of this function
    if len(valid_digits) != 2:
        raise Exception("Error: you must specify exactly 2 digits for classification!")

    data = ds.load_digits()
    labels = data['target']
    features = data['data']

    X = features[(labels == valid_digits[0]) | (labels == valid_digits[1]), :]
    Y = labels[(labels == valid_digits[0]) | (labels == valid_digits[1]),]
    
    X_norm = X/np.max(X, axis=1)[:,None]

    Y[Y == valid_digits[0]] = 0
    Y[Y == valid_digits[1]] = 1

    X_train, X_test, Y_train, Y_test = train_test_split(X_norm, Y, test_size=0.25, random_state=10)
    Y_train = Y_train.reshape((len(Y_train), 1))
    Y_test = Y_test.reshape((len(Y_test), 1))

    return X_train, Y_train, X_test, Y_test

####################################################
## Function 1: Support vector machine  ##
####################################################

def my_SVM(X_train, Y_train, X_test, Y_test, lamb=0.01, num_iterations=200, learning_rate=0.1):
    ## X_train: Training set of features
    ## Y_train: Training set of labels corresponding to X_train
    ## X_test: Testing set of features
    ## Y_test: Testing set of labels correspdonding to X_test
    ## lamb: Regularization parameter
    ## num_iterations: Number of iterations.
    ## learning_rate: Learning rate.

    ## Function learns the parameters of an SVM.
    n = X_train.shape[0]
    p = X_train.shape[1] + 1
    X_train1 = np.concatenate((np.repeat(1, n, axis=0).reshape((n, 1)), X_train), axis=1)
    Y_train = 2 * Y_train - 1
    beta = np.repeat(0., p, axis=0).reshape((p, 1))

    ntest = X_test.shape[0]
    X_test1 = np.concatenate((np.repeat(1, ntest, axis=0).reshape((ntest, 1)), X_test), axis=1)
    Y_test = 2 * Y_test - 1

    acc_train = np.repeat(0., num_iterations, axis=0)
    acc_test = np.repeat(0., num_iterations, axis=0)

    for it in range(num_iterations):
        score = np.dot(X_train1, beta)
        delta = score*Y_train < 1
        dbeta =  np.dot(np.repeat(1, n, axis=0).reshape(1, n), (np.repeat(np.array(delta*Y_train), p).reshape(n,p)*X_train1)/n)
        beta = beta + learning_rate*dbeta.T
        beta[1:,] = beta[1:,] - lamb*beta[1:,]

        acc_train[it] = np.mean(np.sign(score*Y_train))
        acc_test[it] = np.mean(np.sign(np.dot(X_test1, beta)*Y_test))

    ## Function outputs 3 things:
    ## 1. The learned parameters of the SVM, beta
    ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).

    return beta, acc_train, acc_test



######################################
## Function 2: Adaboost ##
######################################
def my_Adaboost(X_train, Y_train, X_test, Y_test, num_iterations=200):
    ## X_train: Training set of features
    ## Y_train: Training set of labels corresponding to X_train
    ## X_test: Testing set of features
    ## Y_test: Testing set of labels correspdonding to X_test
    ## num_iterations: Number of iterations.

    ## Function learns the parameters of an Adaboost classifier.
    n = X_train.shape[0]
    p = X_train.shape[1]
    threshold = 0.8

    X_train1 = 2 * (X_train > threshold) - 1
    Y_train = 2 * Y_train - 1

    X_test1 = 2 * (X_test > threshold) - 1
    Y_test = 2 * Y_test - 1

    beta = np.repeat(0., p).reshape((p, 1))
    w = np.repeat(1. / n, n).reshape((n, 1))

    acc_train = np.repeat(0., num_iterations, axis=0)
    acc_test = np.repeat(0., num_iterations, axis=0)

    for it in range(num_iterations):
        w = w/np.sum(w)
        a = np.dot(np.repeat(1, n, axis=0).reshape(1, n), (np.repeat(np.array(w*Y_train), p).reshape(n,p)*X_train1))
        e = (1-a)/2
        k = np.argmin(e)
        db = .5*np.log((1-e[0,k])/e[0,k])
        beta[k] = beta[k] + db
        w = w*np.exp(-Y_train*X_train1[:,k].reshape(n,1)*db)

        acc_train[it] = np.mean(np.sign(np.dot(X_train1, beta))== Y_train)
        acc_test[it] = np.mean(np.sign(np.dot(X_test1, beta))==Y_test)
    ## Function outputs 3 things:
    ## 1. The learned parameters of the adaboost classifier, beta
    ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
    return beta, acc_train, acc_test
