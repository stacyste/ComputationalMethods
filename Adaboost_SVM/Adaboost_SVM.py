#########################################################
## Stat 202A - Homework 9
## Author: Stephanie Stacy
## Date : 12/6/17
## Description: This script implements a support vector machine, an adaboost classifier
#########################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names,
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to
## double-check your work, but MAKE SURE TO COMMENT OUT ALL
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "os.chdir" anywhere
## in your code. If you do, I will be unable to grade your
## work since Python will attempt to change my working directory
## to one that does not exist.
#############################################################
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

    X = np.asarray(map(lambda k: X[k, :] / X[k, :].max(), range(0, len(X))))

    Y[Y == valid_digits[0]] = 0
    Y[Y == valid_digits[1]] = 1

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)
    Y_train = Y_train.reshape((len(Y_train), 1))
    Y_test = Y_test.reshape((len(Y_test), 1))

    return X_train, Y_train, X_test, Y_test


####################################################
## Function 1: Support vector machine  ##
####################################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Train an SVM to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def my_SVM(X_train, Y_train, X_test, Y_test, lamb=0.01, num_iterations=200, learning_rate=0.1):
    ## X_train: Training set of features
    ## Y_train: Training set of labels corresponding to X_train
    ## X_test: Testing set of features
    ## Y_test: Testing set of labels correspdonding to X_test
    ## lamb: Regularization parameter
    ## num_iterations: Number of iterations.
    ## learning_rate: Learning rate.

    ## Function should learn the parameters of an SVM.


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

    ## Function should output 3 things:
    ## 1. The learned parameters of the SVM, beta
    ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).

    return beta, acc_train, acc_test



######################################
## Function 2: Adaboost ##
######################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Use Adaboost to classify the digits data ##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##

def my_Adaboost(X_train, Y_train, X_test, Y_test, num_iterations=200):
    ## X_train: Training set of features
    ## Y_train: Training set of labels corresponding to X_train
    ## X_test: Testing set of features
    ## Y_test: Testing set of labels correspdonding to X_test
    ## num_iterations: Number of iterations.

    ## Function should learn the parameters of an Adaboost classifier.

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
    ## Function should output 3 things:
    ## 1. The learned parameters of the adaboost classifier, beta
    ## 2. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 3. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
    return beta, acc_train, acc_test

############################################################################
## Testing your functions and visualize the results here##
############################################################################

X_train, Y_train, X_test, Y_test = prepare_data()

####################################################
## Optional examples (comment out your examples!) ##
####################################################

#Here is the code for all of my visualizations on the pdf. I commented it out here
#because I was not sure if we should leave it in, but all of the output can be found
# in the attached file with commentary

#Create the method betas and accuracies
#beta_svm, train_svm, test_svm = my_SVM(X_train, Y_train, X_test, Y_test)
#beta_ada, train_ada, test_ada = my_Adaboost(X_train, Y_train, X_test, Y_test)

#Accuracy plots for SVM
#x = range(200)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_ylim(.9,1.01)
#ax.plot(x, train_svm, color='indianred', linewidth=3)
#ax.plot(x, test_svm, color='slateblue', linewidth=3)
#ax.text(1,1.015, 'Training and Testing Accuracy for SVM')

#Legend patches
#trainpatch = mpatches.Patch(color='indianred', label='Training Accuracy')
#testpatch = mpatches.Patch(color='slateblue', label='Testing Accuracy')
#plt.legend(handles=[trainpatch, testpatch])
#plt.savefig('svm_accuracy.png')
#plt.show()

#Accuracy plots for adaboost
#x = range(200)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_ylim(0.84,.96)
#ax.plot(x, train_ada, color='indianred', linewidth=3)
#ax.plot(x, test_ada, color='slateblue', linewidth=3)
#ax.text(1,.965, 'Training and Testing Accuracy for Adaboost')

#Legend patches
#trainpatch = mpatches.Patch(color='indianred', label='Training Accuracy')
#testpatch = mpatches.Patch(color='slateblue', label='Testing Accuracy')
#plt.legend(handles=[trainpatch, testpatch])
#plt.savefig('adaboost_acc.png')
#plt.show()

#mean of training (and testing) digits data represented as heatmapped 8x8 images (as in original data)
#mean_xtrain_in = np.mean(X_train, axis=0).reshape(8, 8)
#plt.imshow(mean_xtrain_in, cmap=mpl.cm.get_cmap("inferno"))
#plt.show()

#mean_xtrain_svm = np.mean(X_train*np.repeat(beta_svm[1:, :], 272, axis=0).reshape(64, 272).T, axis=0).reshape(8, 8)
#plt.imshow(mean_xtrain_svm, cmap=mpl.cm.get_cmap("inferno"))
#plt.show()

#mean_xtest_svm = np.mean(X_test*np.repeat(beta_svm[1:, :], 91, axis=0).reshape(64, 91).T, axis=0).reshape(8, 8)
#plt.imshow(mean_xtest_svm, cmap=mpl.cm.get_cmap("inferno"))
#plt.show()

#mean_xtrain_ada = np.mean(X_train2*np.repeat(beta_ada, 272, axis=0).reshape(64, 272).T, axis=0).reshape(8, 8)
#plt.imshow(mean_xtrain_ada, cmap=mpl.cm.get_cmap("inferno"))
#plt.show()

#mean_xtest_ada = np.mean(X_test2*np.repeat(beta_ada, 91, axis=0).reshape(64, 91).T, axis=0).reshape(8, 8)
#plt.imshow(mean_xtest_ada, cmap=mpl.cm.get_cmap("inferno"))
#plt.show()
