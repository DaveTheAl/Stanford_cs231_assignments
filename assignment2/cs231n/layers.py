import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # print 'x', x.shape
  # print 'w', w.shape
  # print 'b', b.shape
  tmp_x = np.reshape(x, (x.shape[0], -1))
  out = np.dot(tmp_x, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  tmp_x = np.reshape(x, (x.shape[0], -1))
  #w = np.repeat(w, (x.shape[0]/w.shape[1]), axis=1)
  # print 'x', x.shape
  # print 'w', w.shape
  # print 'b', b.shape
  #the upstream derivative is equivalent to the position at which we derive x, w and b
  dx = np.reshape(np.dot(dout, w.T), x.shape)
  dw = np.dot(tmp_x.T, dout)
  db = np.sum(dout, axis=0)                             #this is equivalent to dot(dout, ones)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  mask = x<=0
  out = np.copy(x)
  out[mask] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  #gradient of x => dx is equal to 0, if max(0, ax) == 0, and to a, if max(0, ax) == x

  mask = x<=0
  dx = np.copy(dout)
  dx[mask] = 0

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.
  
  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################

    #Calculating the mean
    mean = (1.0/N) * np.sum(x, axis=0)

    #Calculating the variance
    upper = x - mean
    variance = (1.0/N) * np.sum(upper**2, axis=0)

    #Calculating x_hat
    lower = np.sqrt(variance + eps)
    x_hat = upper / lower

    #Calculating the output value
    out = gamma * x_hat + beta

    #outputting now
    running_mean = momentum * running_mean + (1.0 - momentum) * mean
    running_var = momentum * running_var + (1.0 - momentum) * variance
    cache = (mean, variance, x_hat, upper, lower, gamma, beta, x, bn_param)      #I don't think that we necessarily need all these values
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################

    norm_x = (x - running_mean) / np.sqrt(running_var + eps)
    out = gamma * norm_x + beta
    cache = (running_mean, running_var, gamma, beta, bn_param)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache


def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.
  
  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """

  #maybe rewrite this with certain conventions...? this is quite messy!


  dx, dgamma, dbeta = None, None, None
  #retrieve values from the cache
  (mean, variance, x_hat, upper, lower, gamma, beta, x, bn_param) = cache
  eps = bn_param.get('eps', 1e-5)
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################

  # xi = x #... how to address the specific x?
  # #3.2 backprop level dlowerdx
  # dlowerdx = 2 * lower * (1-xi)               #xi should be the x we are talking to to....
  # #3.1 backprop level dupperdx
  # dupperdx = 1-xi                             #xi should be the x we are talking to to....
  #
  # #2. backprop level d(x_hat)dx which is equivalent to d(u/v)dx
  # dx_hatdx = (dupperdx * lower - upper * dlowerdx) / lower**2       #maybe should store the internal of lower to avoid numerical errors
  #
  # #1. backprop level dFdx
  # dFdx = gamma * dx_hatdx

  N, D = dout.shape

  dgamma = np.sum(x_hat * dout, axis=0) #dout * x_hat
  dbeta = np.sum(dout, axis=0)

  #calculating these values as described on the paper
  dl_dxhat = gamma * dout #dout * gamma

  #Everything for dl_dvar:
  p1 = x-mean
  p2 = dl_dxhat
  p3 = (lower**2.0)**(-3.0/2)
  p4 = (-1.0/2)
  dl_dvar = np.sum( p1 * p2 * p3 * p4, axis=0)

  #Everything for dl_dmean:
  q1 = dl_dxhat
  q2 = (-1.0 / lower)
  q3 = (1.0/N) * dl_dvar
  q4 = (-2.0 * (x - mean) )
  dl_dmean = np.sum(q1 * q2, axis=0) + q3 * np.sum(q4, axis=0)

  #Everything for dl_dx:
  s1 = dl_dxhat * (1.0 / lower)
  s2 = dl_dvar * (2.0 / N) * (x - mean)
  s3 = dl_dmean * (1.0 / N)
  dl_dx = s1 + s2 + s3

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dx = dl_dx

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.
  
  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.
  
  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.
  
  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  (mean, variance, x_hat, upper, lower, gamma, beta, x, bn_param) = cache
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################
  N, D = dout.shape

  dgamma = np.sum(dout * x_hat, axis=0)
  dbeta = np.sum(dout, axis=0)

  #calculating these values as described on the paper
  dl_dxhat = dout * gamma

  #Everything for dl_dvar:
  p1 = x-mean
  p2 = dl_dxhat
  p3 = (lower**2)**(-3.0/2)
  p4 = (-1.0/2)
  dl_dvar = np.sum( p1 * p2 * p3 * p4, axis=0)

  #Everything for dl_dmean:
  q1 = dl_dxhat
  q2 = (-1.0 / lower)
  q3 = (1.0/N) * dl_dvar
  q4 = (-2.0 * (x - mean) )
  dl_dmean = np.sum(q1 * q2, axis=0) + q3 * np.sum(q4, axis=0)

  #Everything for dl_dx:
  s1 = dl_dxhat * (1.0 / lower)
  s2 = dl_dvar * (2.0 / N) * (x - mean)
  s3 = dl_dmean * (1.0 / N)
  dl_dx = s1 + s2 + s3

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  dx = dl_dx
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  
  return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.randn(x.shape[0], x.shape[1]) < p) / p
    out = mask * x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    mask = None
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################
    dx = dout * mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  P = conv_param['pad']
  S = conv_param['stride']
  out = None
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape


  H_out = 1 + (H + 2*P - HH) / S
  W_out =  1 + (W + 2*P - WW) / S
  out = np.zeros((N, F, H_out, W_out))
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  #print 'Shape of x', x.shape
  npad = ((0,), (0,), (P,), (P, ))
  x_pad = np.pad(x, pad_width=npad, mode='constant')
  N_pad, C_pad, H_pad, W_pad = x_pad.shape

  # print 'Shape of aug_x', aug_x.shape
  # print 'Shape of w', w.shape
  # print 'Shape of out', out.shape

  # height_m = (HH/2.0) #vertical margin (focal scope of concolution)
  # width_m = (WW/2.0) #horizontal margin

  for n in xrange(N):
    for f in xrange(F):
      for i_h in xrange(H_out):
        for i_w in xrange(W_out):                                 #there is a start offset, and regard the stride!
          #select the values around the center of '(w, h)'
          selected_values = x_pad[n, :, i_h*S:i_h*S+HH, i_w*S:i_w*S+WW]        #select all channels...
          # x_pad[n, :, k * S:k * S + HH, l * S:l * S + WW]

          # print 'selected values', selected_values.shape
          # print 'w', w.shape
          # print 'wf', w[f,:,:,:].shape

          #perform the dot product around these values
          dot_prod = np.sum( (selected_values * w[f,:]) )
          res = dot_prod + b[f]
          out[n, f, i_h, i_w] = res  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  (x, w, b, conv_param) = cache
  
  N, F, out_height, out_width = dout.shape
  N, C, H, W = x.shape
  F, C, HH, WW = w.shape

  pad = conv_param['pad']
  stride = conv_param['stride']
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################

  # npad = ((0,0), (0,0), (pad, pad), (pad, pad))
  # aug_x = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
  # aug_dout = np.pad(dout, pad_width=npad, mode='constant', constant_values=0)
  # aug_N, aug_C, aug_H, aug_W = aug_x.shape
  #
  # for n in xrange(N):
  #   for f in xrange(F):
  #     for i_h in xrange(out_height):
  #       for i_w in xrange(out_width):
  #         selected_values = aug_dout[n, :, (i_h*stride):(i_h*stride + HH), (i_w*stride):(i_w*stride + WW) ]
  #         grad = np.sum(selected_values * aug_dout[f, :, :, :])
  #         dw[n, :, (i_h*stride):(i_h*stride + HH), (i_w*stride):(i_w*stride + WW)] += grad

  x, w, b, conv_param = cache
  P = conv_param['pad']
  x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')

  N, C, H, W = x.shape
  F, C, HH, WW = w.shape
  N, F, Hh, Hw = dout.shape
  S = conv_param['stride']

  # For dw: Size (C,HH,WW)
  # Brut force love the loops !
  import sys
  

  dw = np.zeros((F, C, HH, WW))
  for fprime in range(F):
    for cprime in range(C):
      for i in range(HH):
        for j in range(WW):
          sub_xpad = x_pad[:, cprime, i:i + Hh * S:S, j:j + Hw * S:S]
          dw[fprime, cprime, i, j] = np.sum(
                      dout[:, fprime, :, :] * sub_xpad)
          
  # For db : Size (F,)
  db = np.zeros((F))
  for fprime in range(F):
      db[fprime] = np.sum(dout[:, fprime, :, :])

  dx = np.zeros((N, C, H, W))
  for nprime in range(N):
    for i in range(H):
      sys.stdout.write('.')
      for j in range(W):
        for f in range(F):
          for k in range(Hh):
            for l in range(Hw):
              mask1 = np.zeros_like(w[f, :, :, :])
              mask2 = np.zeros_like(w[f, :, :, :])
              if (i + P - k * S) < HH and (i + P - k * S) >= 0:
                mask1[:, i + P - k * S, :] = 1.0
              if (j + P - l * S) < WW and (j + P - l * S) >= 0:
                mask2[:, :, j + P - l * S] = 1.0
              w_masked = np.sum(
                w[f, :, :, :] * mask1 * mask2, axis=(1, 2))
              dx[nprime, :, i, j] += dout[nprime, f, k, l] * w_masked
    print

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  Hp = pool_param['pool_height']
  Wp = pool_param['pool_width']
  S = pool_param['stride']

  N, C, H, W = x.shape
  H1 = (H - Hp) / S + 1
  W1 = (W  - Wp) / S + 1

  out = np.zeros((N, C, H1, W1))
  for n in range(N):
    for ph in range(H1):
      for pw in range(W1):
        for c in range(C):
          out[n, c, ph, pw] = np.max(x[n, c, ph*S:ph*S+Hp, pw*S:pw*S+Wp])

  # out = np.zeros((N, C, (H-p_height)/stride + 1, (W-p_width)/stride + 1))         #should depth be conversed or collided  #plus one for bias?

  # for n in xrange(N):
  #   for im_h in xrange(p_height):
  #     for im_w in xrange(p_width):
  #       for c in xrange(C):
  #         pooling_area = x[n, c, im_h * stride:im_h * stride + p_height, im_w * stride:im_w * stride + p_width]
  #         # print 'Pooling area should be of size (heigth, width)', (p_height, p_width)
  #         # print 'Actual pooling area is of size:', pooling_area.shape
  #         out[n, c, im_h, im_w] = np.max(pooling_area)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  Hp = pool_param['pool_height']
  Wp = pool_param['pool_width']
  S = pool_param['stride']
  N, C, H, W = x.shape
  H1 = (H - Hp) / S + 1
  W1 = (W - Wp) / S + 1

  dx = np.zeros((N, C, H, W))
  for nprime in range(N):
    for cprime in range(C):
      for k in range(H1):
        for l in range(W1):
          x_pooling = x[nprime, cprime, k * S:(k * S) + Hp, l * S:(l * S) + Wp]
          maxi = np.max(x_pooling)
          x_mask = x_pooling == maxi
          dx[nprime, cprime, k * S:(k * S) + Hp, l * S:(l * S) + Wp] += dout[nprime, cprime, k, l] * x_mask

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta
  

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx
