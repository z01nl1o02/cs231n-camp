from builtins import range
from builtins import object
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
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.randn( input_dim, hidden_dim ) * (weight_scale)
        self.params['b1'] = np.zeros( hidden_dim )
        self.params['W2'] = np.random.randn( hidden_dim, num_classes ) * (weight_scale)
        self.params['b2'] = np.zeros( num_classes)
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
        out,self.cache_l1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        out,self.cache_l2 = affine_forward( out, self.params['W2'], self.params['b2'])
        scores = np.copy(out)
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
        loss,grad = softmax_loss(out,y)
        loss += 0.5 * self.reg * (self.params['W1']**2).sum() + 0.5 * self.reg * (self.params['W2'] ** 2).sum()
        
        grad,grads['W2'],grads['b2'] = affine_backward(grad, self.cache_l2)
        grad,grads['W1'],grads['b1'] = affine_relu_backward(grad, self.cache_l1)
        
        grads['W1'] += self.reg * 1.0 * self.params['W1']
        grads['W2'] += self.reg * 1.0 * self.params['W2']
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
import pdb
    
class FCBlock(object):
    def __init__(self, index, input_dim, output_dim, 
                 activation = 'relu', normalization = "batchnorm", 
                 dropout = 1,seed = None, 
                 reg=0.0, weight_scale = 1e-2,dtype=np.float32):
        self.W = np.random.randn( input_dim, output_dim).astype(dtype) * weight_scale
        self.b = np.zeros( output_dim , dtype=dtype)
        self.bn_gamma = np.ones( output_dim, dtype=dtype )
        self.bn_beta = np.zeros( output_dim, dtype=dtype )
        
        self.use_dropout = dropout != 1
        
        self.reg = reg
        self.index = index + 1
        self.caches = {
            'affine':None,
            'norm':None,
            'relu':None,
            'dropout':None
        }
        self.activation = activation
        self.normalization = normalization
        self.bn_param = {"mode":"train"}
        self.dropout_param = {"mode":"train","p":dropout}
        if seed is not None:
            self.dropout_param['seed'] = seed
        
        
        
                
        self.dW = self.W.copy()
        self.db = self.b.copy()
        
        self.dbn_gamma = self.bn_gamma.copy()
        self.dbn_beta = self.bn_beta.copy()
        return 
    
    def set_mode(self, mode):
        if mode == 'train':
            self.dropout_param['mode']='train'
            self.bn_param['mode'] = 'train'
        else:
            self.dropout_param['mode']='test'
            self.bn_param['mode'] = 'test'
        return
    
    def forward(self,X):
        out,self.caches['affine'] = affine_forward(X,self.W,self.b)
        if self.normalization == "batchnorm":
            out,self.caches['norm'] = batchnorm_forward(out,self.bn_gamma, self.bn_beta, self.bn_param)
        if self.activation == 'relu':
            out,self.caches['relu'] = relu_forward(out)
        if self.use_dropout:
            out,self.caches['dropout'] = dropout_forward(out,self.dropout_param)
            
        return out
    
    def backward(self,dout):
        dX = dout.copy()
        if self.use_dropout:
            dX = dropout_backward(dX, self.caches['dropout'])
        if self.activation == 'relu':
            dX = relu_backward(dX, self.caches['relu'])
        if self.normalization == "batchnorm":
            dX,self.dbn_gamma,self.dbn_beta = batchnorm_backward(dX,self.caches['norm'])
        dX,self.dW,self.db = affine_backward(dX,self.caches['affine'])
        self.dW += self.reg * self.W
        return dX
    
    @property
    def reg_val(self):
        return 0.5 * self.reg * (self.W**2).sum()
    
    @property        
    def params(self):
        if self.normalization == "":
            return {
                'W%d'%self.index : self.W,
                'b%d'%self.index : self.b,
            }
        else:
            return {
                'W%d'%self.index : self.W,
                'b%d'%self.index : self.b,
                'gamma%d'%self.index:self.bn_gamma,
                'beta%d'%self.index:self.bn_beta,
            }
    @property
    def grads(self):
        if self.normalization == "":
            return {
                'W%d'%self.index : self.dW,
                'b%d'%self.index : self.db,
            }
        else:
            return {
                'W%d'%self.index : self.dW,
                'b%d'%self.index : self.db,
                'gamma%d'%self.index:self.dbn_gamma,
                'beta%d'%self.index:self.dbn_beta,
            }
    
        
   


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
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
        self.normalization = normalization
        #self.use_dropout = dropout != 1
        #self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}  #dict with all learnable variance

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        
        
        self.layers = []
        prev_dim = input_dim
        dims = map(lambda zz:zz, hidden_dims)
        dims.append( num_classes )
        for k_layer in range(self.num_layers):
            cur_dim = dims[k_layer]
            if k_layer + 1 == self.num_layers:
                layer = FCBlock(k_layer,prev_dim,cur_dim,
                                activation="",normalization = normalization,dropout=1,seed=seed,
                                reg = reg, weight_scale=weight_scale,
                                      dtype=dtype)
            else:
                layer = FCBlock(k_layer,prev_dim,cur_dim,
                                activation="relu",normalization = normalization,dropout=dropout, seed=seed,
                                reg = reg, weight_scale=weight_scale,
                                      dtype=dtype)
            self.layers.append(layer)
            prev_dim = cur_dim
            self.params.update( layer.params )
            
       
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        #self.dropout_param = {}
        #if self.use_dropout:
            #self.dropout_param = {'mode': 'train', 'p': dropout}
            #if seed is not None:
            #    self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        #if self.normalization=='batchnorm':
        #    self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        #for k, v in self.params.items():
        #    self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        #if self.use_dropout:
        #    self.dropout_param['mode'] = mode
        #if self.normalization=='batchnorm':
        #    for bn_param in self.bn_params:
        #        bn_param['mode'] = mode
        scores = None
        
       # print('loss mode {}'.format(mode))
        
        for k_layer in range(self.num_layers):
            self.layers[k_layer].set_mode(mode)
                                                        
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
        out = X.copy()
        for k_layer in range(self.num_layers):
            out = self.layers[k_layer].forward(out)
        scores = out.copy()
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        
        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, grad = softmax_loss(out,y)
        for k_layer in range(self.num_layers-1,-1,-1):
            loss += self.layers[k_layer].reg_val
            grad = self.layers[k_layer].backward(grad)
            grads.update( self.layers[k_layer].grads )
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        #pdb.set_trace()
        return loss, grads
