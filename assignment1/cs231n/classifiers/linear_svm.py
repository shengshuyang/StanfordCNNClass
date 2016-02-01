import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  # initialize variables
  loss = 0.0
  dW = np.zeros(W.shape)
  num_classes = W.shape[1]
  num_train = X.shape[0]

  # calculate margin
  scores = X.dot(W)
  correct_scores = scores[range(num_train),y]
  margin = (scores.T - correct_scores.T).T + 1
  margin = margin.clip(min = 0)
  margin[range(num_train),y] = 0

  # a binary indicator of whether a margin is positive
  activation = (margin > 0).astype(int)

  # sum over all positive activations. the activation of y[i] on X[i] is the
  # negative of the sum
  sum_activation = np.sum(activation, axis = 1)
  activation[range(num_train),y] = -sum_activation

  # it can be proven that dW is simply a dot product
  dW = X.T.dot(activation)
  # and the loss is simply the margin
  loss = np.sum(margin)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss and grad.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW
