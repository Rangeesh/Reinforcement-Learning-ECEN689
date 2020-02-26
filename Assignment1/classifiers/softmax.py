from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        log_probs =X[i].dot(W)
        log_probs -= np.max(log_probs)
        
        probs = np.exp(log_probs)/np.sum(np.exp(log_probs))
        loss -=np.log(probs[y[i]])
        probs[y[i]] -=1
        for j in range(num_classes):
            dW[:,j]+=X[i,:]*probs[j]
            
    
    loss/=num_train
    dW/=num_train
    
    loss+=0.5*reg*np.sum(W*W)
    dW+=reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_classes = W.shape[1]
    num_train = X.shape[0]
    log_probs = np.dot(X,W)
    log_probs -=np.max(log_probs, axis=1, keepdims=True)
    Uprobs=np.exp(log_probs)
    probs = Uprobs/np.sum(Uprobs, axis=1, keepdims=True)
    
    actual_log_probs = -np.log(probs[np.arange(num_train),y])

    loss = np.sum(actual_log_probs) /num_train+0.5*reg*np.sum(W*W)
    
    probs[np.arange(num_train),y]-=1
    dW = np.dot(X.T,probs)
    dW/=num_train
    dW+=reg*W
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
