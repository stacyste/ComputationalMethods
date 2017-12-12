#########################################################
## Stat 202A - Homework 3
## Author: Stephanie Stacy
## Date : October 24
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
#############################################################

##################################
## Function 1: QR decomposition ##
##################################

myQR <- function(A){
  
  ## Perform QR decomposition on the matrix A
  ## Input: 
  ## A, an n x m matrix
  
  n <- nrow(A)
  m <- ncol(A)
  Q <- diag(n)
  R <- A
  
  for(k in 1:(m-1)){
    x <- rep(0,n)
    x[k:n] <- R[c(k:n) , k]
    v <- x
    v[k] <- x[k] + sign(x[k])*sqrt(sum(x^2))
    s <- sqrt(sum(v^2))
    u <- v/s
    
    R <- R - 2*(u%*%t(u)%*%R)
    Q <- Q - 2*(u%*%t(u)%*%Q)
  }
  
  ## Function should output a list with Q.transpose and R
  ## Q is an orthogonal n x n matrix
  ## R is an upper triangular n x m matrix
  ## Q and R satisfy the equation: A = Q %*% R
  return(list("Q" = t(Q), "R" = R))
  
}

###############################################
## Function 2: Linear regression based on QR ##
###############################################

myLM <- function(X, Y){
  n <- nrow(X)
  p <- ncol(X)
  
  Z <- cbind(rep(1,n),X,Y)
  R <- myQR(Z)$R
  R1 <- R[c(1:(p+1)), c(1:(p+1))]
  Y1 <- R[c(1:(p+1)), p+2]
  beta_ls <- solve(R1, Y1)
  
  ## Function returns the 1 x (p + 1) vector beta_ls, 
  ## the least squares solution vector
  return(beta_ls)
  
}

##################################
## Function 3: PCA based on QR  ##
##################################

myEigen_QR <- function(A, numIter = 1000){
  R <- A
  Q <- diag(dim(A)[1])
  
  for(i in 1:numIter){
    decomp <- myQR(R)
    Q <- Q %*% decomp$Q
    R <- decomp$R%*%decomp$Q
  }
  return(list("D" = diag(R), "V" = Q))
}
