#!/usr/bin/env python

import sys
sys.path.append("/u/lambalex/DeepLearning/curriculum")

import numpy.random as rng
import numpy as np

import theano
import theano.tensor as T

import lasagne

from random import randint

from viz import plot_images

import gzip

import cPickle as pickle

#import matplotlib.pyplot as plt

mn = gzip.open("/u/lambalex/data/mnist/mnist.pkl.gz")

train, valid, test = pickle.load(mn)

trainx,trainy = train
validx,validy = valid
testx, testy = test

srng = theano.tensor.shared_randomstreams.RandomStreams(42)

'''
1.  Define a GAN to match a single gaussian.  
2.  Figure out how to make this work and where it breaks.  
    -Discriminator small neural net.  
    -Generator could just be a function of the form mu + sigma*eps where mu and sigma are learned params.  
3.  Next step, add a corruptor of the form inp + sigma*eps.  So it can just inject noise.  Cost is KL-divergence.  


'''


'''
    Just reparam trick.  
'''

def init_params_generator():
    p = {}

    p['w1'] = theano.shared(0.01 * rng.normal(size=(128,512)).astype('float32'))
    p['w2'] = theano.shared(0.01 * rng.normal(size=(512,512)).astype('float32'))
    p['w3'] = theano.shared(0.01 * rng.normal(size=(512,784)).astype('float32'))

    p['b1'] = theano.shared(0.0 * rng.normal(size=(512,)).astype('float32'))
    p['b2'] = theano.shared(0.0 * rng.normal(size=(512,)).astype('float32'))
    p['b3'] = theano.shared(0.0 * rng.normal(size=(784,)).astype('float32'))

    return p

def init_params_corruptor():
    p = {}

    p['w1'] = theano.shared(0.01 * rng.normal(size=(784,200)).astype('float32'))
    p['w2'] = theano.shared(0.01 * rng.normal(size=(200,200)).astype('float32'))
    p['w3'] = theano.shared(0.01 * rng.normal(size=(200,784)).astype('float32'))

    p['b1'] = theano.shared(0.0 * rng.normal(size=(200,)).astype('float32'))
    p['b2'] = theano.shared(0.0 * rng.normal(size=(200,)).astype('float32'))
    p['b3'] = theano.shared(0.0 * rng.normal(size=(784,)).astype('float32'))

    p['log_sigma'] = theano.shared(-1.0 + 0.0*rng.normal(size=(784,)).astype('float32'))

    return p

'''
    Two layer MLP.  
'''
def init_params_discriminator():
    p = {}

    p['w1'] = theano.shared(0.01 * rng.normal(size=(784,200)).astype('float32'))
    p['w2'] = theano.shared(0.01 * rng.normal(size=(200,200)).astype('float32'))
    p['w3'] = theano.shared(0.01 * rng.normal(size=(200,1)).astype('float32'))

    p['b1'] = theano.shared(0.0 * rng.normal(size=(200,)).astype('float32'))
    p['b2'] = theano.shared(0.0 * rng.normal(size=(200,)).astype('float32'))
    p['b3'] = theano.shared(0.0 * rng.normal(size=(1,)).astype('float32'))


    return p

def bn(inp):
    return (inp - T.mean(inp,axis=0)) / (0.001 + T.std(inp,axis=0))

def generator(p,z):
    
    h1 = T.nnet.relu(bn(T.dot(z, p['w1']) + p['b1']), alpha=0.02)
    h2 = T.nnet.relu(bn(T.dot(h1,p['w2']) + p['b2']), alpha=0.02)

    xg = T.dot(h2, p['w3']) + p['b3']

    return xg    

def corruptor(p,x):

    h1 = T.nnet.relu(bn(T.dot(x, p['w1']) + p['b1']), alpha=0.02)
    h2 = T.nnet.relu(bn(T.dot(h1,p['w2']) + p['b2']), alpha=0.02)
    xc = T.dot(h2, p['w3'])*0.0 + p['b3']*0.0 + srng.normal(size=x.shape) * T.exp(p['log_sigma'])

    return xc + x

def discriminator(p,x):
    h1 = T.nnet.relu(bn(T.dot(x, p['w1']) + p['b1']), alpha=0.02)
    h2 = T.nnet.relu(bn(T.dot(h1,p['w2']) + p['b2']), alpha=0.02)
    s = T.dot(h2, p['w3']) + p['b3']

    return s

def clip_params(p,pc):
    newp = []
    for pi in pc:
        if p[pi] in pc:
            print "PARAM 2 CLIP"
            newp.append((pi, T.clip(p[pi], -0.01, 0.01)))
        else:
            newp.append((pi, p[pi]))

    return newp

d_params = init_params_discriminator()
g_params = init_params_generator()
c_params = init_params_corruptor()

x = T.matrix('real_samples')

noise_z = srng.normal(size=(128,128))

x_gen = generator(g_params, noise_z)

x_corr = corruptor(c_params,x)
x_gen_corr = corruptor(c_params,x_gen)

d_real = discriminator(d_params, x_corr)

d_fake = discriminator(d_params, x_gen_corr)

d_loss = d_fake.mean() - d_real.mean()
g_loss = -d_fake.mean()

c_cost = 0.01 * (T.sqr(x_corr - x).mean() + T.sqr(x_gen_corr - x_gen).mean())

lr = 0.001

updates_g = lasagne.updates.rmsprop(g_loss, g_params.values(),learning_rate=lr)

updates_c = lasagne.updates.rmsprop(-d_loss + c_cost, c_params.values(),learning_rate=lr)

updates_g.update(updates_c)

updates_d = clip_params(lasagne.updates.rmsprop(d_loss, d_params.values(),learning_rate=lr), d_params.values())

train_g = theano.function(inputs=[x],outputs=[g_loss,x_gen,x_gen_corr],updates=updates_g)
train_d = theano.function(inputs=[x],outputs=[d_loss,x_corr],updates=updates_d)

do_plot = False

if __name__ == "__main__":

    for iteration in range(0,100000):

        r = randint(0,40000)

        xs = trainx[r:r+128].reshape((128,28*28))

        for k in range(0,5):
            d_loss, x_corr = train_d(xs)

        g_loss, x_gen, x_gen_corr = train_g(xs)

        if iteration % 500 == 0:

            print "=============================================="
            print iteration

            print "c value", c_params['log_sigma'].get_value().mean()

            print "mean match orig", xs.mean(), x_gen.mean()
            print "std match orig", xs.std(), x_gen.std()

            print "mean match corr", x_corr.mean(), x_gen_corr.mean()
            print "std match corr", x_corr.std(), x_gen_corr.std()

            plot_images(xs.reshape((128,1,28,28)), "plots/real.png")
            plot_images(x_gen.reshape((128,1,28,28)), "plots/gen.png")
            plot_images(x_corr.reshape((128,1,28,28)).clip(0.0,1.0), "plots/corrupt_real")
            plot_images(x_gen_corr.reshape((128,1,28,28)).clip(0.0,1.0), "plots/corrupt_gen")

            if do_plot:
                plt.hist(xs,alpha=0.5)
                plt.hist(x_gen,alpha=0.5)
                plt.legend(["real", "fake"])
                plt.show()

                plt.hist(x_corr,alpha=0.5)
                plt.hist(x_gen_corr,alpha=0.5)
                plt.legend(["real corrup", "fake corrup"])
                plt.show()
                                                




