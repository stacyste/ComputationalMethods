/*
#########################################################
## Stat 202A - Homework 4
## Author: Stephanie Stacy
## Date : October 31, 2017
## Description: This script implements QR decomposition,
## and linear regression based on QR.
#########################################################
 
 */ 


# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


/* ~~~~~~~~~~~~~~~~~~~~~~~~~ 
Sign function for later use 
~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
double signC(double d){
  return d<0?-1:d>0? 1:0;
}



/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 1: QR decomposition 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */  

// [[Rcpp::export()]]
List myQRC(const mat A){ 
  
  /*
  Perform QR decomposition on the matrix A
  Input: 
  A, an n x m matrix (mat)
  
  */ 
  int n = A.n_rows;
  int m = A.n_cols;
  int s = 0;
  mat R = A;
  mat Q = eye(n, n);
  mat x(n, 1);
  mat v(n, 1);
  mat u(n, 1);
  List output;
  
  for(int k = 0; k < (m - 1); k++){
    
    x = 0 * x;
    for(int j = k; j < n; j ++){
      x(j, 0) = R(j, k);
    }
    s = -1 * signC(x(k, 0));
    v = x;
    v(k, 0) = x(k, 0) - s * norm(x);
    u = v / norm(v);
    
    R -= 2 * (u * (u.t() * R)); 
    Q -= 2 * (u * (u.t() * Q)); 
    
  }
  
  // Function should output a List 'output', with 
  // Q.transpose and R
  // Q is an orthogonal n x n matrix
  // R is an upper triangular n x m matrix
  // Q and R satisfy the equation: A = Q %*% R
  output["Q"] = Q.t();
  output["R"] = R;
  return(output);
  
  
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 2: Linear regression using QR 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


// [[Rcpp::export()]]
mat myLMC(const mat X, const mat Y){
  
  /*  
  Perform the linear regression of Y on X
  Input: 
  X is an n x p matrix of explanatory variables
  Y is an n dimensional vector of responses
  */
  int p = X.n_cols;
  
  mat Z  = join_rows(X, Y);
  mat R  = myQRC(Z)[1];
  
  mat R1 = R(span(0,p - 1), span(0,p - 1));
  mat Y1 = R(span(0,p - 1), p);

  
  mat beta_ls = inv(R1) * Y1;
  return(beta_ls.t());
  
}  