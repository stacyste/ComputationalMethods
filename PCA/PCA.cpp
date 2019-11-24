/*
#########################################################
## Stat 202A - Homework 3
## Author: Stephanie Stacy
## Date : October 25, 2017
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################
 
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
  
  mat R = A;
  int n = R.n_rows;
  int m = R.n_cols;
  
  mat Q(n,n);
  Q.eye();
  
  for(int k = 0; k < m; k++){
    vec x(n, fill::zeros);
    x(span(k,n-1)) = R(span(k,n-1), k);
    vec v = x;
    v(k) = x(k) + signC(x(k))*norm(x);
    double s = norm(v);
    mat u = v/s;
    R = R - 2*u*u.t()*R;
    Q = Q - 2*u*u.t()*Q;
    
  }
  
  // Function should output a List 'output', with 
  // Q.transpose and R
  // Q is an orthogonal n x n matrix
  // R is an upper triangular n x m matrix
  // Q and R satisfy the equation: A = Q %*% R
  List output;
  output["Q"] = Q.t();
  output["R"] = R;
  return(output);
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 2: Linear regression using QR 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */
// [[Rcpp::export()]]
mat myLinearRegressionC(const mat X, const mat Y){
  
  /* 
  Perform the linear regression of Y on X
  Input: 
  X is an n x p matrix of explanatory variables
  Y is an n dimensional vector of responses
  Do NOT simulate data in this function. n and p
  should be determined by X.
  Use myQRC inside of this function
  */  
  int n = X.n_rows;
  int p = X.n_cols;
  
  mat intercept = ones(n,1);
  mat Z = join_rows(intercept, X);
  Z = join_rows(Z, Y);
  
  List output = myQRC(Z);
  mat R = output(1);
  mat R1 = R(span(0,p), span(0,p));
  mat Y1 = R(span(0,p), p+1);
  mat beta_ls = inv(R1)*Y1;
  
  // Function returns the 'p+1' by '1' matrix 
  // beta_ls of regression coefficient estimates
  return(beta_ls.t());
  
}  


/* ~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 3: PCA based on QR 
~~~~~~~~~~~~~~~~~~~~~~~~~~ */


// [[Rcpp::export()]]
List myEigen_QRC(const mat A, const int numIter = 1000){
  /*  
  Perform PCA on matrix A using your QR function, myQRC.
  Input:
  A: Square matrix
  numIter: Number of iterations
  */  
  
  mat R = A;
  int n = R.n_rows;
  mat Q(n,n);
  Q.eye();
  
  for(int i = 0; i < numIter; i++){
    List decomp = myQRC(R);
    mat d_Q = decomp(0);
    mat d_R = decomp(1);
    Q = Q*d_Q;
    R = d_R*d_Q;
  }
  
  vec D = R.diag();
  // Function should output a list with D and V
  // D is a vector of eigenvalues of A
  // V is the matrix of eigenvectors of A (in the 
  // same order as the eigenvalues in D.)
  List output;
  output["D"] = D;
  output["V"] = Q;
  return(output);
  
}

