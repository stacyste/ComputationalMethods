#########################################################
## Stat 202A - Final Project
## Author: Stephanie Stacy
## Date : 12/11/2017
## Description: This script implements a two layer neural network in Tensorflow
#########################################################

############################################################################
## Implement a two layer neural network in Tensorflow to classify MNIST digits ##
############################################################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Train a two layer neural network to classify the MNIST dataset ##
## Relu as the activation function for the first layer. 
## Softmax as the activation function for the second layer
## z=Relu(x*W1+b1) ##
## y=Softmax(z*W2+b2)##
# Cross-entropy as the loss function
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    #create the model - specify input x placeholder, output layer placeholder and initialize the weights between
    x = tf.placeholder(tf.float32, [None, 784])

    # first set of weights and bias
    W1 = tf.Variable(tf.random_normal([784, 100], stddev=.1))
    b1 = tf.Variable(tf.random_normal([100], stddev = .1))

    # second set of weights and bias
    W2 = tf.Variable(tf.random_normal([100, 10], stddev=.1))
    b2 = tf.Variable(tf.random_normal([10], stddev = .1))

    # run relu on the hidden layer
    z = tf.nn.relu(tf.matmul(x, W1) + b1)
    y = tf.matmul(z, W2) + b2


    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    #the optimizer
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
