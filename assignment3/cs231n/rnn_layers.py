import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
  """
  Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
  activation function.

  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.

  Inputs:
  - x: Input data for this timestep, of shape (N, D).
  - prev_h: Hidden state from previous timestep, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)

  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - cache: Tuple of values needed for the backward pass.
  """
  next_h, cache = None, None
  ##############################################################################
  # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
  # hidden state and any values you need for the backward pass in the next_h   #
  # and cache variables respectively.                                          #
  ##############################################################################
  #For a single timestep
  #x should be a one-hot vector!!

  new_input = np.dot(x, Wx)
  rec_input = np.dot(prev_h, Wh)

  next_h = np.tanh(new_input + rec_input + b)
  cache = (x, prev_h, Wx, Wh, b, new_input, rec_input)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return next_h, cache


def rnn_step_backward(dnext_h, cache):
  """
  Backward pass for a single timestep of a vanilla RNN.
  
  Inputs:
  - dnext_h: Gradient of loss with respect to next hidden state
  - cache: Cache object from the forward pass
  
  Returns a tuple of:
  - dx: Gradients of input data, of shape (N, D)
  - dprev_h: Gradients of previous hidden state, of shape (N, H)
  - dWx: Gradients of input-to-hidden weights, of shape (N, H)
  - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
  - db: Gradients of bias vector, of shape (H,)
  """
  dx, dprev_h, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
  #                                                                            #
  # HINT: For the tanh function, you can compute the local derivative in terms #
  # of the output value from tanh.                                             #
  ##############################################################################
  x, prev_h, Wx, Wh, b, new_input, rec_input = cache


  dout = (1- np.tanh(new_input + rec_input + b)**2) * dnext_h #bcs this is the gradient of tanh

  dx = np.dot(dout, Wx.T) #not sure what to multiply with...
  dWx = np.dot(x.T, dout)  #not sure what to multiply with...

  dprev_h = np.dot(dout, Wh.T) #maybe multiply with //* np.ones(x.shape)
  dWh = np.dot(prev_h.T, dout)

  db = np.sum(dout, axis=0)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
  """
  Run a vanilla RNN forward on an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The RNN uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the RNN forward, we return the hidden states for all timesteps.
  
  Inputs:
  - x: Input data for the entire timeseries, of shape (N, T, D).
  - h0: Initial hidden state, of shape (N, H)
  - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
  - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
  - b: Biases of shape (H,)
  
  Returns a tuple of:
  - h: Hidden states for the entire timeseries, of shape (N, T, H).
  - cache: Values needed in the backward pass
  """
  h, cache = None, None
  ##############################################################################
  # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
  # input data. You should use the rnn_step_forward function that you defined  #
  # above.                                                                     #
  ##############################################################################
  N, T, D = x.shape
  N, H = h0.shape
  h = np.zeros((N, T, H))
  cache = []

  h[:, -1, :] = h0  #takes care of t==0 : h_prev = h0... //I hope the rest takes care of the -1...
  for i in xrange(T):
    #print
    #print "Debugging..."
    #print np.sum(abs(h0 - h[:, i-1, :]) / h0)
    #rnn_step_forward(x, prev_h, Wx, Wh, b)
    h[:,i,:], c = rnn_step_forward(x[:, i, :], h[:, i-1, :], Wx, Wh, b)
    cache.append(c)

  #print
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return h, cache


def rnn_backward(dh, cache):
  """
  Compute the backward pass for a vanilla RNN over an entire sequence of data.
  
  Inputs:
  - dh: Upstream gradients of all hidden states, of shape (N, T, H)
  
  Returns a tuple of:
  - dx: Gradient of inputs, of shape (N, T, D)
  - dh0: Gradient of initial hidden state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
  - db: Gradient of biases, of shape (H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  ##############################################################################
  # TODO: Implement the backward pass for a vanilla RNN running an entire      #
  # sequence of data. You should use the rnn_step_backward function that you   #
  # defined above.                                                             #
  ##############################################################################
  #N, T, D = x.shape
  #N, H = h0.shape

  N, T, H = dh.shape
  D = cache[0][0].shape[1]

  dx = np.zeros((N, T, D), dtype=float)
  dWx = np.zeros((D, H), dtype=float)
  dh0 = np.zeros((N, H), dtype=float)      #RNN parameters seem to be constant....
  db = np.zeros((H), dtype=float)
  dWh = np.zeros((H, H), dtype=float) #HH is correct.. maybe transpose...?

  dh_prev = np.zeros((N, H))
  for i in reversed(xrange(T)):
    dh_current = dh[:, i, :] + dh_prev
    t_dx, dh_prev, t_dWx, t_dWh, t_db = rnn_step_backward(dh_current, cache[i])

    #dynamic vectors:
    dx[:, i, :] += t_dx        #multiplication by dh handled within ght step function! / i think should be = instead of +=
    dh0 = dh_prev    #actual recurrent element

    #things im sure of (bcs shared parameters)
    dWh += t_dWh      #what the fuck????!!!!
    dWx += t_dWx
    db += t_db

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
  """
  Forward pass for word embeddings. We operate on minibatches of size N where
  each sequence has length T. We assume a vocabulary of V words, assigning each
  to a vector of dimension D.
  
  Inputs:
  - x: Integer array of shape (N, T) giving indices of words. Each element idx
    of x muxt be in the range 0 <= idx < V.
  - W: Weight matrix of shape (V, D) giving word vectors for all words.
  
  Returns a tuple of:
  - out: Array of shape (N, T, D) giving word vectors for all input words.
  - cache: Values needed for the backward pass
  """
  out, cache = None, None
  ##############################################################################
  # TODO: Implement the forward pass for word embeddings.                      #
  #                                                                            #
  # HINT: This should be very simple.                                          #
  ##############################################################################
  # N, T = x.shape
  # V, D = W.shape
  # out = np.zeros((N, T, D))
  #print x
  #print W
  # out[: ,:, :] = W[np.arange(N), np.arange(T), x]

  # for n in xrange(N):
  #   for t in xrange(T):
  #     idx = x[n, t]
  #     out[n, t, :] = W[idx, :]

  out = W[x, :]

  cache = (x, W)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return out, cache


def word_embedding_backward(dout, cache):
  """
  Backward pass for word embeddings. We cannot back-propagate into the words
  since they are integers, so we only return gradient for the word embedding
  matrix.
  
  HINT: Look up the function np.add.at
  
  Inputs:
  - dout: Upstream gradients of shape (N, T, D)
  - cache: Values from the forward pass
  
  Returns:
  - dW: Gradient of word embedding matrix, of shape (V, D).
  """
  dW = None
  ##############################################################################
  # TODO: Implement the backward pass for word embeddings.                     #
  #                                                                            #
  # HINT: Look up the function np.add.at                                       #
  ##############################################################################
  x, W = cache
  N, T = x.shape
  V, D = W.shape
  dW = np.zeros((V, D))

  np.add.at(dW, x, dout)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  return dW


def sigmoid(x):
  """
  A numerically stable version of the logistic sigmoid function.
  """
  pos_mask = (x >= 0)
  neg_mask = (x < 0)
  z = np.zeros_like(x)
  z[pos_mask] = np.exp(-x[pos_mask])
  z[neg_mask] = np.exp(x[neg_mask])
  top = np.ones_like(x)
  top[neg_mask] = z[neg_mask]
  return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
  """
  Forward pass for a single timestep of an LSTM.
  
  The input data has dimension D, the hidden state has dimension H, and we use
  a minibatch size of N.
  
  Inputs:
  - x: Input data, of shape (N, D)
  - prev_h: Previous hidden state, of shape (N, H)
  - prev_c: previous cell state, of shape (N, H)
  - Wx: Input-to-hidden weights, of shape (D, 4H)
  - Wh: Hidden-to-hidden weights, of shape (H, 4H)
  - b: Biases, of shape (4H,)
  
  Returns a tuple of:
  - next_h: Next hidden state, of shape (N, H)
  - next_c: Next cell state, of shape (N, H)
  - cache: Tuple of values needed for backward pass.
  """
  next_h, next_c, cache = None, None, None
  #############################################################################
  # TODO: Implement the forward pass for a single timestep of an LSTM.        #
  # You may want to use the numerically stable sigmoid implementation above.  #
  #############################################################################
  D, H = Wx.shape
  H = H/4 #bcs Wx includes parameters for all a..

  # print("Wx shape is: ", Wx.shape)
  # print("Wh shape is: ", Wh.shape)
  # print("b shape is: ", b.shape)
  # m1 = np.concatenate((Wx, Wh, b[:, np.newaxis]), axis=0)
  # m2 = np.concatenate((x, prev_h, np.ones(like(b.T)) ), axis=1)
  # a = np.dot(m1, m2.T)
  #is also a possibility

  a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  
  a_i = a[:, 0*H:1*H]
  a_f = a[:, 1*H:2*H]
  a_o = a[:, 2*H:3*H]
  a_g = a[:, 3*H:4*H]

  i = sigmoid(a_i)
  f = sigmoid(a_f)
  o = sigmoid(a_o) #there is a numerical faster way if we first concatenate the matrices i think
  g = np.tanh(a_g)

  t1 = (f * prev_c)
  t2 = (i * g)
  next_c = (t1 + t2)

  t3 = np.tanh(next_c)
  next_h = (o * t3)

  cache = (next_h, next_c, x, prev_h, prev_c, Wx, Wh, b, a_i, a_f, a_o, a_g, i, f, o, g)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
  """
  Backward pass for a single timestep of an LSTM.
  
  Inputs:
  - dnext_h: Gradients of next hidden state, of shape (N, H)
  - dnext_c: Gradients of next cell state, of shape (N, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data, of shape (N, D)
  - dprev_h: Gradient of previous hidden state, of shape (N, H)
  - dprev_c: Gradient of previous cell state, of shape (N, H)
  - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for a single timestep of an LSTM.       #
  #                                                                           #
  # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
  # the output value from the nonlinearity.                                   #
  #############################################################################
  next_h, next_c, x, prev_h, prev_c, Wx, Wh, b, a_i, a_f, a_o, a_g, i, f, o, g = cache

  #calculating the local gradients:
  #next_h = (o * t3)
  #t3 = np.tanh(next_c)
  dnext_c += o * (1 - np.tanh(next_c)**2) * dnext_h
  do = np.tanh(next_c) * dnext_h

  #next_c = (t1 + t2)
  #t2 = (i * g)
  di = g * dnext_c
  dg = i * dnext_c
  #t1 = (f * prev_c)
  df = prev_c * dnext_c
  dprev_c = f * dnext_c


  #g = np.tanh(a_g)
  da_g = ( 1-(np.tanh(a_g)**2) ) * dg  #derivative of tanh
  #o = sigmoid(a_o) #there is a numerical faster way if we first concatenate the matrices i think
  da_o = o*(1-o)*do   #derivative of sigmoid
  #f = sigmoid(a_f)
  da_f = f*(1-f)*df   #derivative of sigmoid
  #i = sigmoid(a_i)
  da_i = i*(1-i)*di   #derivative of sigmoid

  # a_g = a[:, 3*H:4*H]
  # a_o = a[:, 2*H:3*H]
  # a_f = a[:, 1*H:2*H]
  # a_i = a[:, 0*H:1*H]
  
  #a = np.dot(x, Wx) + np.dot(prev_h, Wh) + b
  da = np.hstack((da_i, da_f, da_o, da_g))

  dx = np.dot(da, Wx.T)
  dWx = np.dot(x.T, da)

  dprev_h = np.dot(da, Wh.T)
  dWh = np.dot(prev_h.T, da)

  db = np.sum(da, axis=0)

  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
  """
  Forward pass for an LSTM over an entire sequence of data. We assume an input
  sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
  size of H, and we work over a minibatch containing N sequences. After running
  the LSTM forward, we return the hidden states for all timesteps.
  
  Note that the initial cell state is passed as input, but the initial cell
  state is set to zero. Also note that the cell state is not returned; it is
  an internal variable to the LSTM and is not accessed from outside.
  
  Inputs:
  - x: Input data of shape (N, T, D)
  - h0: Initial hidden state of shape (N, H)
  - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
  - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
  - b: Biases of shape (4H,)
  
  Returns a tuple of:
  - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
  - cache: Values needed for the backward pass.
  """
  h, cache = None, None
  #############################################################################
  # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
  # You should use the lstm_step_forward function that you just defined.      #
  #############################################################################
  N, T, D = x.shape
  N, H = h0.shape
  
  prev_c = np.zeros(h0.shape)

  h = np.zeros((N, T, H))
  cache = []
  h[:, -1, :] = h0  #takes care of t==0 : h_prev = h0... //I hope the rest takes care of the -1...
  
  for i in xrange(T):
    #print
    #print "Debugging..."
    #print np.sum(abs(h0 - h[:, i-1, :]) / h0)
    #rnn_step_forward(x, prev_h, Wx, Wh, b)
    h[:, i, :], prev_c, c = lstm_step_forward(x[:, i, :], h[:, i-1, :], prev_c, Wx, Wh, b)
    cache.append(c)
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################

  return h, cache


def lstm_backward(dh, cache):
  """
  Backward pass for an LSTM over an entire sequence of data.]
  
  Inputs:
  - dh: Upstream gradients of hidden states, of shape (N, T, H)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient of input data of shape (N, T, D)
  - dh0: Gradient of initial hidden state of shape (N, H)
  - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
  - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
  - db: Gradient of biases, of shape (4H,)
  """
  dx, dh0, dWx, dWh, db = None, None, None, None, None
  #############################################################################
  # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
  # You should use the lstm_step_backward function that you just defined.     #
  #############################################################################
  N, T, H = dh.shape
  next_h, next_c, x, prev_h, prev_c, Wx, Wh, b, a_i, a_f, a_o, a_g, i, f, o, g = cache[0]
  D = x.shape[-1]

  dx = np.zeros((N, T, D), dtype=float)
  dh0 = np.zeros((N, H), dtype=float)      #RNN parameters seem to be constant....
  dWx = np.zeros((D, 4*H), dtype=float)
  db = np.zeros((4*H), dtype=float)
  dWh = np.zeros((H, 4*H), dtype=float) #HH is correct.. maybe transpose...?

  dprev_h = np.zeros((N, H))
  dprev_c = np.zeros((N, H))
  for i in reversed(xrange(T)):
    dh_current = dh[:, i, :] + dprev_h
    dc_current = dprev_c
    t_dx, dprev_h, dprev_c, t_dWx, t_dWh, t_db = lstm_step_backward(dh_current, dc_current, cache[i])
    #t_dx, dh_prev, t_dWx, t_dWh, t_db = rnn_step_backward(dh_current, cache[i])

    #dynamic vectors:
    dx[:, i, :] += t_dx        #multiplication by dh handled within ght step function! / i think should be = instead of +=
    dh0 = dprev_h    #actual recurrent element

    #things im sure of (bcs shared parameters)
    dWh += t_dWh      #what the fuck????!!!!
    dWx += t_dWx
    db += t_db
  ##############################################################################
  #                               END OF YOUR CODE                             #
  ##############################################################################
  
  return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
  """
  Forward pass for a temporal affine layer. The input is a set of D-dimensional
  vectors arranged into a minibatch of N timeseries, each of length T. We use
  an affine function to transform each of those vectors into a new vector of
  dimension M.

  Inputs:
  - x: Input data of shape (N, T, D)
  - w: Weights of shape (D, M)
  - b: Biases of shape (M,)
  
  Returns a tuple of:
  - out: Output data of shape (N, T, M)
  - cache: Values needed for the backward pass
  """
  N, T, D = x.shape
  M = b.shape[0]
  out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
  cache = x, w, b, out
  return out, cache


def temporal_affine_backward(dout, cache):
  """
  Backward pass for temporal affine layer.

  Input:
  - dout: Upstream gradients of shape (N, T, M)
  - cache: Values from forward pass

  Returns a tuple of:
  - dx: Gradient of input, of shape (N, T, D)
  - dw: Gradient of weights, of shape (D, M)
  - db: Gradient of biases, of shape (M,)
  """
  x, w, b, out = cache
  N, T, D = x.shape
  M = b.shape[0]

  dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
  dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
  db = dout.sum(axis=(0, 1))

  return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
  """
  A temporal version of softmax loss for use in RNNs. We assume that we are
  making predictions over a vocabulary of size V for each timestep of a
  timeseries of length T, over a minibatch of size N. The input x gives scores
  for all vocabulary elements at all timesteps, and y gives the indices of the
  ground-truth element at each timestep. We use a cross-entropy loss at each
  timestep, summing the loss over all timesteps and averaging across the
  minibatch.

  As an additional complication, we may want to ignore the model output at some
  timesteps, since sequences of different length may have been combined into a
  minibatch and padded with NULL tokens. The optional mask argument tells us
  which elements should contribute to the loss.

  Inputs:
  - x: Input scores, of shape (N, T, V)
  - y: Ground-truth indices, of shape (N, T) where each element is in the range
       0 <= y[i, t] < V
  - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
    the scores at x[i, t] should contribute to the loss.

  Returns a tuple of:
  - loss: Scalar giving loss
  - dx: Gradient of loss with respect to scores x.
  """

  N, T, V = x.shape
  
  x_flat = x.reshape(N * T, V)
  y_flat = y.reshape(N * T)
  mask_flat = mask.reshape(N * T)
  
  probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
  dx_flat = probs.copy()
  dx_flat[np.arange(N * T), y_flat] -= 1
  dx_flat /= N
  dx_flat *= mask_flat[:, None]
  
  if verbose: print 'dx_flat: ', dx_flat.shape
  
  dx = dx_flat.reshape(N, T, V)
  
  return loss, dx

