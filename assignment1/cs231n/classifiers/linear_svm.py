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
  for i in range(num_train):#for every train example
    scores = X[i].dot(W)    #calculate the score for every class
    correct_class_score = scores[y[i]]#the correct class score
    for j in range(num_classes):#for every class in this i example
      if j == y[i]:         #for the correct class we don't calculate the loss to have a minimum loss of Zero
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1 #check if the diff between score and correct score < margin = 1
      if margin > 0:        #if diff < margin there is a loss
        loss += margin      #update the total loss
        dW[:,j] += X[i]     #update dWj by +X[i] in case margin > 0
        dW[:,y[i]] -= X[i]  #update dwyi by -X[i] in case margin > 0
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
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

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  scores = X.dot(W)     #calculate the score for each class (N,C)
  correct_class_scores = scores[range(num_train),y]     #get the correct class score for each input (N,1)
  
  margin = scores - correct_class_scores.reshape(num_train,1) + 1    #get the margin between all classes and the correct one (N,C)
  margin[margin < 0 ] = 0           # only get the margin when > 0
  margin[range(num_train), y] = 0   #remove the value at j==yi
  loss = np.sum(margin)             #sum all the margins to get total loss
  loss /= num_train                 #get average
  loss += .5 * reg * np.sum(W*W)    #add the regularization coefficient

  # (wj*xi -wyi*xi+1)
  activations = np.ones(margin.shape,dtype=int)*(margin > 0) # get all positions where dX will be updated
  num_activations_sample = np.sum(activations, axis = 1)
#  print(num_activations_sample)
  activations[range(num_train), y] = -num_activations_sample # for each time a dwj is calculated dwyi has to be subtracted so foreach train sample we calculate how dw calculated so dyi be subtracted the same amount
  dW = X.T.dot(activations)
#  print(dW)
  dW /= num_train
  dW += reg * W

  #pass
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
  #pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
