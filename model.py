from theano.tensor.nnet.conv import conv2d
from theano import tensor as T
import theano
import numpy as np


# Convert to floatX to keep theano happy
def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


# Weights are initialization uniformly in [-0.01, 0.01]
def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))


# Biases are initialized to 0
def init_biases(shape):
    b_values = np.zeros((shape[0],), dtype=theano.config.floatX)
    return theano.shared(value=b_values, borrow=True, name="Conv_b")


# RMSprop is used for updates
def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates


# RELU activation function
def rectify(X):
    return T.maximum(X, 0.)


# Optional dropout for hidden layers
def dropout(X, srng, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X


# Setup the network, given initial parameters
def model(X, filter_params, bias_params, p_drop_conv, srng):
    inp = X
    half = int(len(filter_params)/2)
    conv_params = filter_params[:half]
    deconv_params = filter_params[half:]
    conv_biases = bias_params[:half]
    deconv_biases = bias_params[half:]
    for f, b in zip(conv_params, conv_biases):
        outa = rectify(conv2d(inp, f, border_mode='valid') +
                       b.dimshuffle('x', 0, 'x', 'x'))
        outb = dropout(outa, srng, p_drop_conv)
        inp = outb
    c = 0
    for f, b in zip(deconv_params, deconv_biases):
        if c == len(deconv_params):
            outa = T.nnet.sigmoid(conv2d(inp, f, border_mode='full') +
                                  b.dimshuffle('x', 0, 'x', 'x'))
        else:
            outa = rectify(conv2d(inp, f, border_mode='full') +
                           b.dimshuffle('x', 0, 'x', 'x'))
        outb = dropout(outa, srng, p_drop_conv)
        inp = outb
        c += 1
    output = inp
    return output


# Setup the parameters for the network
def get_params(img_x, filters):
    filter_params = []
    bias_params = []
    # convolution layers:
    for f in filters:
        w = init_weights(f)
        filter_params.append(w)
        b = init_biases(f)
        bias_params.append(b)
    i = 0
    # deconvolution layers:
    for f in reversed(filters):
        if i == len(filters)-1:
            w = init_weights((1, f[0], f[2], f[3]))
            b = init_biases((1, f[0], f[2], f[3]))
        else:
            w = init_weights((f[1], f[0], f[2], f[3]))
            b = init_biases((f[1], f[0], f[2], f[3]))
        filter_params.append(w)
        bias_params.append(b)
        i += 1
    return filter_params, bias_params
