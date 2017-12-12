/*
####################################################
## Stat 202A - Homework 2
## Author: Stephanie Stacy
## Date : October 14, 2017
## Description: This script implements linear regression 
## using the sweep operator
####################################################
 
###########################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not change your working directory
## anywhere inside of your code. If you do, I will be unable 
## to grade your work since R will attempt to change my 
## working directory to one that does not exist.
###########################################################
 
 */ 


# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;



/* ~~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 1: Sweep operator 
~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
mat mySweepC(const mat A, int m){
  mat B = A;
  int n = B.n_rows;
  for(int k = 0; k < m; k++){
    for(int i = 0; i < n; i ++){
      for(int j = 0; j < n; j ++){
        if(i != k && j !=k){
          B(i,j) = B(i,j) - B(i,k)*B(k,j)/B(k,k);
        }
      }
    }
    double pivot = B(k,k);
    B.row(k) = B.row(k)/pivot;
    B.col(k) = B.col(k)/pivot;
    B(k,k) = -1/pivot;
  }
  return(B);
  
}


/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 2: Linear regression using the sweep operator 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


// [[Rcpp::export()]]
mat myLinearRegressionC(const mat X, const mat Y){
  
  /*  
  Find the regression coefficient estimates beta_hat
  corresponding to the model Y = X * beta + epsilon
  Your code must use the sweep operator you coded above.
  Note: we do not know what beta is. We are only 
  given a matrix X and a matrix Y and we must come 
  up with an estimate beta_hat.
  
  X: an 'n row' by 'p column' matrix of input variables.
  Y: an 'n row' by '1 column' matrix of responses*/
  
  int n = X.n_rows;
  int p = X.n_cols;
  
  mat intercept = ones(n,1);
  mat Z = join_rows(intercept, X);
  Z = join_rows(Z, Y);
  mat A = Z.t()*Z;
  mat B = mySweepC(A, p+1);
  mat beta_hat = B(span(0,p), p+1);
  
  
  // Function returns the 'p+1' by '1' matrix 
  // beta_hat of regression coefficient estimates
  return(beta_hat);
  
}


