import numpy as np
from random import shuffle

###########################################################
# loss = -log( softmax(x) )
#
#
#
###########################################################

def delta_softmax(softmax_values,idx):
    val = softmax_values[idx]
    softmax_values = -1*softmax_values * val
    softmax_values[idx] = val*(1-val)
    return softmax_values
    
    

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
  D,C = W.shape
  N,_ = X.shape
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_pred = X.dot(W)
  y_pred_exp = np.exp(y_pred)
    
  #avoid numeric instability
  y_pred_exp_norm = np.tile( np.reshape(y_pred_exp.max(axis=1),(N,1)), (1,C))
  y_pred_exp = y_pred_exp / y_pred_exp_norm
  
  y_pred_exp_sum = np.tile( np.reshape(y_pred_exp.sum(axis=1),(N,1)), (1,C))
  y_pred_softmax = y_pred_exp / y_pred_exp_sum
 
  for batch_idx in range(N):
        gt = int(y[batch_idx])
        loss += -np.log(y_pred_softmax[batch_idx,gt])
        delta = -1*delta_softmax(y_pred_softmax[batch_idx], gt) / y_pred_softmax[batch_idx,gt]
       # print dW.shape
       # print X[batch_idx].shape
       # print delta.shape
        dW += np.reshape(X[batch_idx],(-1,1)).dot(np.reshape(delta,(1,-1)))
        
  loss /= N          
  loss += reg * (W**2).sum()    
    
  
  dW /= N
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
    
  D,C = W.shape
  N,_ = X.shape
    
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_pred = X.dot(W)
  y_pred_exp = np.exp(y_pred)
    
  #avoid numeric instability
  y_pred_exp_norm = np.tile( np.reshape(y_pred_exp.max(axis=1),(N,1)), (1,C))
  y_pred_exp = y_pred_exp / y_pred_exp_norm
  
  y_pred_exp_sum = np.tile( np.reshape(y_pred_exp.sum(axis=1),(N,1)), (1,C))
  y_pred_softmax = y_pred_exp / y_pred_exp_sum
    
  
  loss = (-np.log(y_pred_softmax[np.arange(N),y])).mean()
  #loss = -np.log(loss)  
  loss += reg * (W**2).sum()
  
    
  delta = y_pred_softmax
  #for k in range(N):
  #      gt = int(y[k])
  #      delta[k,gt] -= 1
  delta[np.arange(N),y] = y_pred_softmax[np.arange(N),y] - 1
    
  #for n in range(N):
  #  x = np.reshape( X[n,:], (-1,1))
  #  d = np.reshape( delta[n,:], (1,-1))
  #  dW += x.dot(d)
  #dW_list = map(lambda (x,d): np.reshape(x,(-1,1)).dot(np.reshape(d,(1,-1))), zip(X.tolist(), delta.tolist()))
  #dW = reduce(lambda x,y: x+y, dW_list) / N
  dW = X.transpose().dot(delta)
  dW /= N
  dW += 2 * reg * W
    
    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

