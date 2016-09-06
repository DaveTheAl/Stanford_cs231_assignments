import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_dim  = X.shape[1]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #y[y == 10] = 0                                                                    #Set all samples of 10 to 0
  for i in xrange(num_train):
    #for c in xrange(num_classes):
    M = np.dot(X[i,:], W)
    # print 'M', M.shape
    stability_term = np.max(M)                                                        #For numerical stability
    M += stability_term
    # print 'M2', M.shape
    all_probs = np.exp(M) / np.sum( np.exp(M) )   #Computing probability
    loss += -np.log(all_probs[y[i]])                                             #Computing loss
    # print 'all_probs', all_probs.shape
    # print 'y', y[i]
    #dW[i] = all_probs[y[i]] - y[i]

    #added from other repo
    all_probs[y[i]] -= 1 # subracting 1 when classes match                            #why this???
    dW += np.dot(np.reshape(X[i],(num_dim,1)), np.reshape(all_probs,(1,num_classes))) #... hm ... what does this line do...?

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= float(num_train)
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_dim  = X.shape[1]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #First calculating the loss
  M = np.dot(X, W)
  stability_term = np.max(M)                                                        #For numerical stability
  M += stability_term
  all_probs = np.exp(M) / np.sum( np.exp(M), axis=1 )[:, np.newaxis]   #Computing probability
  loss += np.sum(-np.log(all_probs[np.arange(num_train), y]))      #hope this gives the elemtnwise loss...


  #Second, calculating the weight differences...
  all_probs[np.arange(num_train),y] -= 1
  # print 'X', X.shape
  # print 'all_probs', all_probs.shape
  dW += np.dot(X.T, all_probs)

  #regularization for loss, and for weigths
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dW /= float(num_train)
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

