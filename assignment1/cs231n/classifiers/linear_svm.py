import numpy as np
from random import shuffle

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
    #grad_i = np.zeros(W.shape)
    #initialize gradient vector
    num_changes = 0                                                             #ADDED
    for j in xrange(num_classes):
      if j == y[i]:
        #dW[:,j] = 0                                                            #ADDED
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:]
        num_changes += 1
        #grad_i[W*X[i,:] - correct_class_score + 1 > 0, j] = -X[i,:]            #ADDED
        #num_changes += 1                                                       #ADDED
    dW[np.arange(dW.shape[0]),y[i]] -= num_changes * X[i,:]                     #ADDED  bcs we add the average gradient

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #Add regularization to the loss
  dW = (1.0/num_train)*dW + reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """

  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  M_scores = np.dot(X, W)
  correct_class_score = M_scores[np.arange(num_train), y];

  # print 'M_scores', M_scores.shape
  # print 'correct class score', correct_class_score.shape
  margin = M_scores - correct_class_score[:, np.newaxis] + 1.0
  margin[np.arange(num_train), y] = 0
  loss = np.sum(margin[margin > 0])
  loss /= float(num_train)
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #incorrect_counts = np.sum(margin > 0, axis=1)            #number of incorrect counts

  # wj = np.sum(X[margin > 0], axis = 0)
  # firstterm = incorrect_counts[y == incorrect_counts]
  # secondterm = X[np.arange(num_train), y]
  # wy = firstterm[:, np.newaxis] * secondterm
  # wy = np.sum(wy, axis=0)
  #
  # dW = wj + wy
  # #dW -= margin[margin > 0]


  # Fully vectorized version. Roughly 10x faster.
  X_mask = np.zeros(margin.shape)
  # column maps to class, row maps to sample; a value v in X_mask[i, j]
  # adds a row sample i to column class j with multiple of v
  X_mask[margin > 0] = 1
  # for each sample, find the total number of classes where margin > 0
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  dW /= num_train # average out weights
  dW += reg*W # regularize the weights


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
