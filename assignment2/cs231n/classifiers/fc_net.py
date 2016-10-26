import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################

    self.params['W1'] = np.random.normal(scale=weight_scale, size=(input_dim, hidden_dim))
    self.params['W2'] = np.random.normal(scale=weight_scale, size=(hidden_dim, num_classes))
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['b2'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    z1, cache_z1 = affine_forward(X, self.params['W1'], self.params['b1'])
    a2, cache_a2 = relu_forward(z1);
    #could also use affine_relu_forward
    # a2 = affine_forward(X, self.params['W1'], self.params['b1'])
    z2, cache_z2 = affine_forward(a2, self.params['W2'], self.params['b2'])

    #beggining of the softmax function
    if y is None:
      scores = z2
      # scores = np.exp(X - np.max(x, axis=1, keepdims=True))
      # scores /= np.sum(scores, axis=1, keepdims=True)
    # else:
      #apply the softmax loss function directly if y is not None
      # scores, dz2  = softmax_loss(z2, y)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    #Adding
    loss, da2 = softmax_loss(z2, y)
    da2, grads['W2'], grads['b2'] = affine_backward(da2, cache_z2)         #where dz1 is equal to dw!
    dz1 = relu_backward(da2, cache_a2)
    dx, grads['W1'], grads['b1'] = affine_backward(dz1, cache_z1)

    #Adding regularization to all terms
    loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2'] ))
    grads['W1'] += self.reg * self.params['W1']
    grads['W2'] += self.reg * self.params['W2']

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    #make a random cell and distribute it over all layer values instead of a for loop

    #{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    self.L = len(hidden_dims) + 1
    self.N = input_dim
    self.C = num_classes

    dims = [input_dim] + hidden_dims + [num_classes]

    Ws = {'W' + str(i + 1):
              weight_scale * np.random.randn(dims[i], dims[i + 1]) for i in range(len(dims) - 1)}
    b = {'b' + str(i + 1): np.zeros(dims[i + 1]) for i in range(len(dims) - 1)}

    self.params.update(b)
    self.params.update(Ws)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    #vorsicht... vorsicht... das wird schiefgehen, aber bisdahin abwarten
    # print
    # print "HEYHO!"
    # print type(input_dim)
    # print type(num_classes)
    # print type(hidden_dims[0])
    # print len(hidden_dims)
    # print self.num_layers

    # L = self.num_layers
    # self.input_dim = input_dim
    # self.num_classes = num_classes
    # dim_arr = [input_dim] + hidden_dims + [num_classes]
    #
    # for i in xrange(1, L-1):          #because counting starts from 0
    #   self.params['W%d' % i] = np.random.randn(dim_arr[i], dim_arr[i+1]) * weight_scale
    #   self.params['b%d' % i] = np.zeros(dim_arr[i+1])
        #initialization, so no need to highly optimized...  (putting if away from for loop


    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = {'bn_param' + str(i+1): {'mode': 'train',
                                                'running_mean': np.zeros(dims[i + 1]),
                                                'running_var': np.zeros(dims[i + 1])
                                                }
                        for i in xrange(len(dims) - 2)}
      gamma = {'gamma' + str(i+1): np.ones(dims[i + 1]) for i in xrange(len(dims) - 2)}
      beta = {'beta' + str(i+1): np.ones(dims[i + 1]) for i in xrange(len(dims) - 2)}

      self.params.update(beta)
      self.params.update(gamma)
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
    # the class scores for X and storing them in the scores variable.          #
    #                                                                          #
    # When using dropout, you'll need to pass self.dropout_param to each       #
    # dropout forward pass.                                                    #
    #                                                                          #
    # When using batch normalization, you'll need to pass self.bn_params[0] to #
    # the forward pass for the first batch normalization layer, pass           #
    # self.bn_params[1] to the forward pass for the second batch normalization #
    # layer, etc.                                                              #
    ############################################################################


    #Second try guys


    hidden = {}
    hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))

    for i in range(self.L):
      idx = i + 1
      # Naming of the variable
      w = self.params['W' + str(idx)]
      b = self.params['b' + str(idx)]
      h = hidden['h' + str(idx - 1)]

      if idx == self.L:
        h, cache_h = affine_forward(h, w, b)
        hidden['h' + str(idx)] = h
        hidden['cache_h' + str(idx)] = cache_h
      else:
        if self.use_batchnorm:
          gamma = self.params['gamma' + str(idx)]
          beta = self.params['beta' + str(idx)]
          bn_param = self.bn_params['bn_param' + str(idx)]

          h, cache_h = affine_norm_relu_forward(h, w, b, gamma, beta, bn_param)
          hidden['h' + str(idx)] = h
          hidden['cache_h' + str(idx)] = cache_h
        else:
          h, cache_h = affine_relu_forward(h, w, b)
          hidden['h' + str(idx)] = h
          hidden['cache_h' + str(idx)] = cache_h

    scores = hidden['h' + str(self.L)]

    #{affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    # out_a = [None] * self.num_layers
    # out_r = [None] * self.num_layers
    # cache_a = [None] * self.num_layers
    # cache_r = [None] * self.num_layers
    # arr_a = ()
    # arr_cache = ()
    # dx_a = [None] * self.num_layers
    # dx_r = [None] * self.num_layers
    #
    # L = self.num_layers
    # out = [None] * self.num_layers    #check how many times we will actually propagate this through
    # cache = [None] * self.num_layers


    # layer = {}
    # layer[0] = X
    # cache_layer = {}
    #
    # for i in xrange(1, self.num_layers):
    #   layer[i], cache_layer[i] = affine_relu_forward(layer[i - 1],
    #                                                  self.params['W%d' % i],
    #                                                  self.params['b%d' % i])
    # #

    #forward propagating through all gates
    #it now feels right.....
    # print 'Getting into the stuff'
    # hidden = {}
    # hidden['h0'] = X.reshape(X.shape[0], np.prod(X.shape[1:]))
    #
    # for i in xrange(L):
    #   idx = i + 1
    #   w = self.params['W%d' % (idx)]
    #   b = self.params['b%d' % (idx)]
    #   h = hidden['h' + str(idx - 1)]
    #   if idx == L:
    #     h, cache_h = affine_forward(h, w, b)
    #     hidden['h' + str(idx)] = h
    #     hidden['cache_h' + str(idx)] = cache_h
    #   else:
    #     h, cache_h = affine_relu_forward(h, w, b)
    #     hidden['h' + str(idx)] = h
    #     hidden['cache_h' + str(idx)] = cache_h



    # out[0] = np.reshape(X, (X.shape[0], np.prod(X.shape[1:])) )
    # cache[0] = [None]
    # print L
    # for i in xrange(L-1):        #should this be only this many times? THis should be 2, because indexing starts at 0, and we have num_layers-1 layers in the hidden unit
    #   out[i+1], cache[i+1] = affine_relu_forward(out[i], self.params['W%d' % i], self.params['b%d' % i])
    #   print 'These two must be compatible'
    #   print self.params['W%d' % i].shape
    #   print self.params['W%d' % (i+1)].shape
    #   print 'These two must be compatible'
    #   print self.params['b%d' % i].shape
    #   print self.params['b%d' % (i+1)].shape
    #
    # out[L], cache[L] = affine_forward(out[L-2], self.params['W%d' % (L-1)], self.params['W%d' % (L-1)])   #L-1 because we start counting at 0


    # out_a[0], cache_a[0] = affine_forward(X, self.params['W%d' % 0], self.params['b%d' % 0])
    # out_r[0], cache_r[0] = relu_forward(out_a[0])

    # print "HEY"
    # for i in xrange(1, self.num_layers-1):
    #   print i
    #   out_a[i], cache_a[i] = affine_forward(out_r[i-1], self.params['W%d' % (i)], self.params['b%d' % (i)])
    #   out_r[i], cache_r[i] = relu_forward(out_a[i])
    #
    # out_a[self.num_layers],  cache_a[self.num_layers] = affine_forward(out_a[self.num_layers - 1], self.params['W%d' % (self.num_layers)], self.params['b%d' % (self.num_layers)])  #supposed to be the very last forward propagation


    # WLast = 'W%d' % self.num_layers
    # bLast = 'b%d' % self.num_layers
    # scores, cache_scores = affine_forward(layer[-1],
    #                                       self.params[WLast],
    #                                       self.params[bLast])

    #softmax loss if y is not none
    # scores = out_a[self.num_layers]
    if y is None:
      return scores

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    # loss, dx_a[self.num_layers] = softmax_loss(scores, y)     #second value should be dout
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################

    #Second try guys

    # Computing of the loss
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0
    for w in [self.params[f] for f in self.params.keys() if f[0] == 'W']: reg_loss += 0.5 * self.reg * np.sum(w * w)

    loss = data_loss + reg_loss

    hidden['dh' + str(self.L)] = dscores
    for i in range(self.L)[::-1]:
      idx = i + 1
      dh = hidden['dh' + str(idx)]
      h_cache = hidden['cache_h' + str(idx)]
      if idx == self.L:
        dh, dw, db = affine_backward(dh, h_cache)
        hidden['dh' + str(idx - 1)] = dh
        hidden['dW' + str(idx)] = dw
        hidden['db' + str(idx)] = db
      else:
        if self.use_batchnorm:

          dh, dw, db, dgamma, dbeta= affine_norm_relu_backward(dh, h_cache)
          hidden['dh' + str(idx - 1)] = dh
          hidden['dW' + str(idx)] = dw
          hidden['db' + str(idx)] = db
          hidden['dgamma' + str(idx)] = dgamma
          hidden['dbeta' + str(idx)] = dbeta

        else:
          dh, dw, db = affine_relu_backward(dh, h_cache)
          hidden['dh' + str(idx - 1)] = dh
          hidden['dW' + str(idx)] = dw
          hidden['db' + str(idx)] = db

    list_dw = {key[1:]: val + self.reg * self.params[key[1:]] for key, val in hidden.iteritems() if key[:2] == 'dW'}
    # Paramerters b
    list_db = {key[1:]: val for key, val in hidden.iteritems() if key[:2] == 'db'}
    list_dgamma = {key[1:]: val for key, val in hidden.iteritems() if key[:6] == 'dgamma'}
    # Paramters beta
    list_dbeta = {key[1:]: val for key, val in hidden.iteritems() if key[:5] == 'dbeta'}

    grads = {}
    grads.update(list_dw)
    grads.update(list_db)
    grads.update(list_dgamma)
    grads.update(list_dbeta)

    #forward propagating through all gates
    # out_a[0], cache_a[0] = affine_forward(X, self.params['W' + str(0)], self.params['b' + str(0) ])
    # out_r[0], cache_r[0] = relu_forward(out_a[0])
    #
    # for i in xrange(1, self.num_layers-1):
    #   out_a[i], cache_a[i] = affine_forward(out_a[i-1], self.params['W' + str(i)], self.params['b' + str(i)])
    #   out_r[i], cache_r[i] = relu_forward(out_a[i])
    #
    # out_a[self.num_layers],  cache_a[self.num_layers] = affine_forward(out_a[self.num_layers - 1], self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])  #supposed to be the very last forward propagation
    #
    # #softmax loss if y is not none
    # scores = out_a[self.num_layers]


    #should it be dx_a[self.num_layers] or dx_a[self.num_layers - 1]
    # dx_a[self.num_layers-1], grads['W%d' % (self.num_layers)], grads['b%d' % (self.num_layers)] = affine_forward(dx_a[self.num_layers], cache_a[self.num_layers])
    #
    # for i in reversed(xrange(self.num_layers - 1)):
    #   dx_r[i] = relu_backward(dx_a[i], cache_r[i])
    #   dx_a[i], grads['W%d' % (i)], grads['b%d' % (i)] = affine_backward(dx_r[i], cache_a[i])


    # dx_a[self.num_layers], dw[self.num_layers], db[self.num_layers] = affine_backward(dout, cache_a[self.num_layers])   #suppsed to be the very first backward propagation
    #
    # grads['W'][self.num_layers] = dw[self.num_layers]
    # for i in reversed(xrange(self.num_layers-1)):
    #   dx_r[i] = relu_backward(dx_a[i+1], cache_a[i])
    #   dx_a[i], dw[i], db[i] = affine_backward(dx_r[i], cache_r[i])
    #   grads['W'][i] = dw[i]
    #   grads['b'][i] = db[i]
    # ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    # for i in xrange(self.num_layers):
    #   loss += 0.5 * self.reg * (np.sum(self.params['W%d' % (i)]))

    ############################################################################
    #                           ACTUAL END OF YOUR CODE                        #
    ############################################################################

    return loss, grads


def affine_norm_relu_forward(x, w, b, gamma, beta, bn_param):
  a1, affine_cache = affine_forward(x, w, b)
  a2, batchnorm_cache = batchnorm_forward(a1, gamma, beta, bn_param)
  out, relu_cache = relu_forward(a2)
  cache = (affine_cache, batchnorm_cache, relu_cache)
  return out, cache

def affine_norm_relu_backward(dout, cache):
  affine_cache, batchnorm_cache, relu_cache = cache
  dx2 = relu_backward(dout, relu_cache)
  dx1, dgamma, dbeta = batchnorm_backward(dx2, batchnorm_cache)
  dx, dw, db = affine_backward(dx1, affine_cache)
  return dx, dw, db, dgamma, dbeta
