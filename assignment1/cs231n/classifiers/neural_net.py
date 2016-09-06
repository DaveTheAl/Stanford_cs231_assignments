import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)


  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # print 'X', X.shape
    # print 'W1', W1.shape
    # print 'b1', b1.shape
    # print 'W2', W2.shape
    # print 'b2', b2.shape
    #print 'y', y.shape
    num_train = X.shape[0]

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    #input (X) - fully connected layer (a1) - ReLU (a2) - fully connected layer (a3) - softmax (z)
    # a1 = np.concatenate(np.dot(X, W1), b1)                #first fully connected layer
    # a2 = svm_loss_vectorized(W1, X, 0, 0)               #output of ReLU
    # #a3 = np.concatenate(np.dot(a2_input, W2), b2)         #second fully connected layer
    # z = softmax_loss_vectorized(W2, a2, 0, 0)             #should include the class scores for the input

    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1) # pass through ReLU activation function
    scores = np.dot(a1, W2) + b2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
    #First calculating the loss

    # compute the class probabilities
    stability_term = np.amax(scores)
    all_probs = np.exp(scores + stability_term) / np.sum(np.exp(scores + stability_term), axis=1)[:, np.newaxis]
    correct = -np.log(all_probs[np.arange(num_train), y])
    loss = np.sum(correct) / float(num_train)
    loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)

    #Second, calculating the weight differences...
    # all_probs[np.arange(num_train),y] -= 1
    # print 'X', X.shape
    # print 'all_probs', all_probs.shape
    # dW += np.dot(X.T, all_probs)

    #regularization for loss, and for weigths
    # loss /= num_train
    # loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

    # dW /= float(num_train)
    # dW += reg*W
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################


    # compute the gradient on scores
    delta_scores = all_probs
    delta_scores[np.arange(num_train),y] -= 1
    delta_scores /= num_train

    # Between Layer 2 and output! (Matrix W2!)
    grads['W2'] = np.dot(a1.T, delta_scores)
    grads['b2'] = np.sum(delta_scores, axis=0)


    # hidden layer
    delta_hidden = np.dot(delta_scores, W2.T)
    # ReLU
    delta_hidden[a1 <= 0] = 0

    # backprop onto Layer 1 (Matrix W1!)
    grads['W1'] = np.dot(X.T, delta_hidden)
    grads['b1'] = np.sum(delta_hidden, axis=0)

    # regularization
    grads['W2'] += reg * W2
    grads['W1'] += reg * W1


    #backpropagating on the second weight set

    # dW2 = np.zeros(W2.shape)
    # dW1 = np.zeros(W1.shape)
    #
    # #SOFTMAX LAYER
    # M = np.copy(scores)
    # stability_term = np.amax(M, axis=1)                                                        #For numerical stability
    # M += stability_term
    # all_probs = np.exp(M) / np.sum( np.exp(M), axis=1 )[:, np.newaxis]   #Computing probability
    # all_probs[np.arange(num_train),y] -= 1
    # #following the exampl delta = cost_derivative(x, y) .* softmax_derivative(z_input_to_this_layer)
    # #all_probs[np.arange(num_train),y] -= 1 ### this equals the softmax_derivative(z_input_to_this_layer)
    # #np.dot(X.T, all_probs) ### this equals the cost_derivative(x, y)
    # delta_2 =  np.dot(X.T, all_probs)
    #
    # delta_1 = np.dot()

    #derivative of ReLU
    #if (max(0, a(x)) == 0):
    #   0
    #else:
    #   da(x)/dx

    #derivative of Softmax
    #all_probs = np.exp(M) / np.sum( np.exp(M), axis=1 )[:, np.newaxis]
    #all_probs[np.arange(num_train),y] -= 1
    #dW += np.dot(X.T, all_probs)
    #
    # all_probs[np.arange(num_train),y] -= 1
    # print 'X', X.shape
    # print 'all_probs', all_probs.shape
    # dW += np.dot(X.T, all_probs)


    #ReLU LAYER
    # correct_class_score = scores[np.arange(num_train), y];
    #
    # # print 'M_scores', M_scores.shape
    # # print 'correct class score', correct_class_score.shape
    # margin = scores - correct_class_score[:, np.newaxis] + 1.0
    # margin[np.arange(num_train), y] = 0
    #
    # X_mask = np.zeros(margin.shape)
    # # column maps to class, row maps to sample; a value v in X_mask[i, j]
    # # adds a row sample i to column class j with multiple of v
    # X_mask[margin > 0] = 1
    # # for each sample, find the total number of classes where margin > 0
    # incorrect_counts = np.sum(X_mask, axis=1)
    # X_mask[np.arange(num_train), y] = -incorrect_counts
    # dW = X.T.dot(X_mask)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_iters):
      # X_batch = None
      # y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      #applying this mask to choose the batches...
      mask = np.random.choice(np.arange(num_train), batch_size)
      X_batch = X[mask]
      y_batch = y[mask]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['b2'] -= learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = np.mean(self.predict(X_batch) == y_batch)
        val_acc = np.mean(self.predict(X_val) == y_val)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    z1 = np.dot(X, self.params['W1']) + self.params['b1']
    a1 = np.maximum(0, z1) # pass through ReLU activation function
    scores = np.dot(a1, self.params['W2']) + self.params['b2']
    y_pred = np.argmax(scores, axis=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


