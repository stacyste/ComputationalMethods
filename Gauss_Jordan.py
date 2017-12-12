# -*- coding: utf-8 -*-
"""

 Stat 202A - Homework 1
 Author: Stephanie Stacy
 Date : October 10, 2017
 Description: This script implements linear regression 
 using Gauss-Jordan elimination in both plain and
 vectorized forms

 INSTRUCTIONS: Please fill in the missing lines of code
 only where specified. Do not change function names, 
 function inputs or outputs. You can add examples at the
 end of the script (in the "Optional examples" section) to 
 double-check your work, but MAKE SURE TO COMMENT OUT ALL 
 OF YOUR EXAMPLES BEFORE SUBMITTING.

 Do not use any of Python's built in functions for matrix 
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

def myLinearRegression(X, Y):
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
  


########################################################
## Optional examples (comment out before submitting!) ##
########################################################

## def testing_Linear_Regression():
  
  ## This function is not graded; you can use it to 
  ## test out the 'myLinearRegression' function 

  ## You can set up a similar test function as was 
  ## provided to you in the R file.