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
#  ## first Implementation
#  loss = 0.0
#  dW = np.zeros_like(W)
#  num_classes = W.shape[1]
#  num_train = X.shape[0]
#  #num_Dim = W.shape[0]
#  
#  for sampleIndex in range(num_train):
#      scores = np.dot(X[sampleIndex,],W)
#      loss += -np.log(np.exp(scores[y[sampleIndex]])/np.exp(scores).sum())
#  
#  loss /= num_train  
#  loss += reg * np.sum(W * W)
  
  
  ### Second Implementations
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  num_Dim = W.shape[0]
  scores = []
  for sampleIndex in range(num_train):#iterate over each train example
      cscores = []
      for cIndex in range(num_classes):#iterate over each class in train example
          cscore = 0.0
          for wIndex in range(num_Dim):#iterate over each Dim in class weights
              #print (cscore)
              cscore += X[sampleIndex, wIndex]*W[wIndex,cIndex]#w*input
          cscores.append(cscore)#add up the results for each class for each sample
      scores.append (cscores)
      
      
  scores = np.array(scores)
  #print(scores.shape)
  for sampleIndex in range(num_train):#calculate the losses over each sample
      loss += -np.log(np.exp(scores[sampleIndex,y[sampleIndex]])/np.exp(scores[sampleIndex,]).sum())
  
  loss /= num_train
  for cIndex in range(num_classes):# add reg to the loss
      for wIndex in range(num_Dim):
          loss += reg * W[wIndex, cIndex]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

