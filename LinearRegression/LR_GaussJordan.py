# -*- coding: utf-8 -*-
"""

 Stat 202A - Homework 1
 Author: Stephanie Stacy
 Date : October 10, 2017
 Description: This script implements linear regression 
 using Gauss-Jordan elimination in both plain and
 vectorized forms

 Does not use any of Python's built in functions for matrix 
 inversion or for linear modeling (except for debugging or 
 in the optional examples section).
 
"""

import numpy as np

###############################################
## Function 1: Plain version of Gauss Jordan ##
###############################################
def myGaussJordan(A, m):
    n = A.shape[0]
    B = np.hstack((A, np.identity(n)))
    for k in range(m):
        a = B[k,k]
        for j in range(2*n):
            B[k,j] = B[k,j]/a
        for i in range(n):
            if i != k:
                b=B[i,k]
                for j in range(2*n):
                    B[i,j] = B[i,j]-B[k,j]*b
    return B
  


####################################################
## Function 2: Vectorized version of Gauss Jordan ##
####################################################

def myGaussJordanVec(A, m):
    n = A.shape[0]
    B = np.hstack((A, np.identity(n)))
    for k in range(m):
        B[k,:] = B[k,:]/B[k,k]
        for i in range(n):
            if i != k:  
                B[i,:] = B[i,:]-B[k,:]*B[i,k]
  ## Function returns the np.array B
    return B


######################################################
## Function 3: Linear regression using Gauss Jordan ##
######################################################
def myLinearRegressionGJ(X, Y):
    n = X.shape[0]
    p = X.shape[1]
    Y = Y.reshape(n,1) #if y i a vector transform it to an nx1 array
    intercept = np.ones((n,1))
    
    Z = np.hstack((intercept, X, Y)) 
    Z_transpose = np.transpose(Z)
    
    A = np.dot(Z_transpose, Z)
    B = myGaussJordanVec(A, p+1)
    beta_hat = B[:(p+1), p+1]
    return beta_hat
