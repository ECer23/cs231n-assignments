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

  #############################################################################
  # TODO                                                                      #
  # Compute the softmax loss and its gradient using explicit loops.           #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    shifted_scores = scores - np.max(scores)
    correct_class_score = shifted_scores[y[i]]
    loss += np.log(np.sum(np.exp(shifted_scores))) - correct_class_score
    for j in range(num_classes):
      softmax_out = np.exp(shifted_scores[j]) / np.sum(np.exp(shifted_scores))
      if j != y[i]:
        dW[:, j] += softmax_out * X[i]
      else:
        dW[:, y[i]] += softmax_out * X[i] - X[i]

  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
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

  #############################################################################
  # TODO                                                                      #
  # Compute the softmax loss and its gradient using no explicit loops.        #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, num_features = X.shape
  scores = X.dot(W)
  shifted_scores = scores - np.max(scores)
  correct_class_scores = np.choose(y, shifted_scores.T)
  loss += np.mean(np.log(np.sum(np.exp(shifted_scores), axis=1)) - correct_class_scores)
  loss += reg * np.sum(W * W)
  softmax_out = np.exp(shifted_scores) / np.sum(np.exp(shifted_scores), axis=1).reshape(-1,1)
  softmax_out[np.arange(num_train), y] -= 1
  dW += X.T.dot(softmax_out)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

