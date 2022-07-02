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
    dim = W.shape[1]
    score = np.dot(X,W)
    for i in range(num_train):
        cur_score = score[i,:]
        normalized_score = cur_score - np.max(cur_score)
        loss = loss + (-normalized_score[y[i]] + np.log(np.sum(np.exp(normalized_score))))
        #print(normalized_score[8],normalized_score.shape[0])
        for j in range(dim):
            softmax_scores = np.exp(normalized_score[j])/np.sum(np.exp(normalized_score))
            if y[i]==j:
                dW[:,j] = dW[:,j] - X[i,:] + softmax_scores * X[i,:]
            else:
                dW[:,j] = dW[:,j] + softmax_scores * X[i,:]
    loss = (loss / num_train) + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W
    pass

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
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    score = np.dot(X,W)
    shift_score = score - np.max(score, axis = 1)[:,np.newaxis]
    print(shift_score.shape)
    correct_score = np.choose(y,shift_score.T)
    current_loss = -correct_score + np.log(np.sum(np.exp(shift_score),axis = 1))
    loss = np.sum(current_loss) / num_train + reg * np.sum(W * W)
    soft_score = np.exp(shift_score)/np.sum(np.exp(shift_score),axis = 1)[:,np.newaxis]
    soft_score[range(num_train), y] -= 1
    dW = (np.dot(X.T, soft_score))/num_train + 2 * reg * W


    
    
    
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
