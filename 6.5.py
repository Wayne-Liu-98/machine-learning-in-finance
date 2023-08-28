# %% [markdown]
# <a href="https://colab.research.google.com/gist/jteichma/f0df299304472502462555a438ea29e6/lsv_calibration.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
#%tensorflow_version 1.x

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
#import tensorflow as tf

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Concatenate, Dropout, Subtract, \
                        Flatten, MaxPooling2D, Multiply, Lambda, Add, Dot
from keras.backend import constant
from keras import optimizers

from keras.models import Model
from keras.layers import Input
from keras import initializers
from keras.constraints import max_norm
import keras.backend as K

# %% [markdown]
# First we create two identical models where either only hedges or local volatilities can be trained.

# %%
m = 10 # layer dimension
n = 2 # number of layers for local volatility
N = 20 # time discretization (should fit to maturities)
maturities = [0.5, 1.] # list of maturities in years

T = 1.0

layers = []
for j in range(len(maturities)):
    layersatT = []
    for i in range(n):
        if i < 1:
            nodes = m
        else:
            nodes = 1
        layer = Dense(nodes, activation='relu', trainable=False,
                      kernel_initializer=initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                      bias_initializer='random_normal')
        layersatT = layersatT + [layer]
    layers = layers + [layersatT]

#P = {(1.0,1.0): 0.4, (1.1,1.0):0.2, (0.9,1.0):0.5, 
#     (1.0,0.5): 0.2, (1.1,0.5):0.1, (0.9,0.5):0.3}

P= {(0.9, 0.5): 0.20042534,
    (0.9, 1.0): 0.23559685,
    (1.0, 0.5): 0.16312157,
    (1.0, 1.0): 0.20771958,
    (1.1, 0.5): 0.13154241,
    (1.1, 1.0): 0.18236567}

hedges = {}
hedgeskey =[]
for key in P.keys():
    for j in range(N):
        hedge = Dense(nodes, activation='relu', trainable=True,
                      kernel_initializer=initializers.RandomNormal(0,0.1),#kernel_initializer='random_normal',
                      bias_initializer='random_normal')
        hedgeskey = hedgeskey + [hedge]
    hedges[key] = hedgeskey
start = 0

# %%
keylist = list(P.keys())
price = Input(shape=(1,))
hedgepf = [Input(shape=(1,)) for l in range(len(P.keys()))]
inputs = [price] + hedgepf
inputshelper = []
hedgeratio = {}
hedge = {}
pricekey = [0 for l in range(len(P.keys()))]

normal = tf.distributions.Normal(loc=0., scale=1.)

def BS(x):
    price=x[0]
    vola=x[1]
    return normal.cdf((K.log(K.abs(price)/key[0])-0.5*(key[1]-j*T/N)*vola**2)/(0.00001+np.sqrt(key[1]-j*T/N)*vola))
# increases computational time

for i in range(len(maturities)):
    for j in range(start,N):
        if maturities[i] >= j*T/N:
            helper0 = layers[i][0](price)
            for k in range(1,2):
                helper0 = layers[i][k](helper0) # local vol applied to price at time j*T/N
            BMincr = Input(shape=(1,)) # BM increment
            stochvol = Input(shape=(1,)) # stochvol value
            helper1 = Multiply()([helper0,BMincr])
            helper1 = Lambda(lambda x: x * np.sqrt(T/N))(helper1)
            priceincr = Multiply()([helper1,stochvol]) # new price increment
            for l in range(len(P.keys())):
                key = keylist[l]
                hedgeratio[key] = hedges[key][j](price)
                BSstrategy = Lambda(BS)([price,helper0])
                hedgeratio[key] = Add()([hedgeratio[key],BSstrategy])
                hedge[key] = Multiply()([priceincr,hedgeratio[key]])
                hedgepf[l] = Add()([hedgepf[l],hedge[key]])
                if key[1]==((j+1)*T/N): # the option expires
                    helper2 = Lambda(lambda x : 0.5*(abs(x-key[0])+x-key[0]))(price)
                    helper2 = Subtract()([helper2,hedgepf[l]]) # payoff minus hedge 
                    pricekey[l] = helper2
            price = Add()([price,priceincr]) #new price after one time step
            inputshelper = inputshelper + [stochvol]
            inputs = inputs + [BMincr]
        else:
            start = j
            break

inputs = inputs + inputshelper
pricekey = Concatenate()(pricekey)
localvol_trainhedge = Model(inputs=inputs, outputs=pricekey)

# %% [markdown]
# Pay attention here: don't jump back and forth when defining models since layer lists come with the same names, so the order of execution is important!

# %%
layers = []

for j in range(len(maturities)):
    layersatT = []
    for i in range(2):
        if i < 1:
            nodes = m
        else:
            nodes = 1
        layer = Dense(nodes, activation='relu', trainable=True,
                      kernel_initializer=initializers.RandomNormal(0,1),#kernel_initializer='random_normal',
                      bias_initializer='random_normal')
        layersatT = layersatT + [layer]
    layers = layers + [layersatT]


hedges = {}
hedgeskey =[]
for key in P.keys():
    for j in range(N):
        hedge = Dense(nodes, activation='relu', trainable=False,
                      kernel_initializer=initializers.RandomNormal(0,0.1),#kernel_initializer='random_normal',
                      bias_initializer='random_normal')
        hedgeskey = hedgeskey + [hedge]
    hedges[key] = hedgeskey
start = 0

# %%
keylist = list(P.keys())
price = Input(shape=(1,))
hedgepf = [Input(shape=(1,)) for l in range(len(P.keys()))]
inputs = [price] + hedgepf
inputshelper = []
hedgeratio = {}
hedge = {}
pricekey = [0 for l in range(len(P.keys()))]

for i in range(len(maturities)):
    for j in range(start,N):
        if maturities[i] >= j*T/N:
            layers[i][0].trainable=True
            helper0 = layers[i][0](price)
            for k in range(1,2):
                layers[i][k].trainable=True
                helper0 = layers[i][k](helper0)
            BMincr = Input(shape=(1,))
            stochvol = Input(shape=(1,))
            helper1 = Multiply()([helper0,BMincr])
            helper1 = Lambda(lambda x: x * np.sqrt(T/N))(helper1)
            priceincr = Multiply()([helper1,stochvol])
            for l in range(len(P.keys())):
                key = keylist[l]
                hedges[key][j].trainable=False
                hedgeratio[key] = hedges[key][j](price)
                BSstrategy = Lambda(BS)([price,helper0])
                hedgeratio[key] = Add()([hedgeratio[key],BSstrategy])
                hedge[key] = Multiply()([priceincr,hedgeratio[key]])
                hedgepf[l] = Add()([hedgepf[l],hedge[key]])
                if key[1]==((j+1)*T/N):
                    helper2 = Lambda(lambda x : 0.5*(abs(x-key[0])+x-key[0]))(price)
                    helper2 = Subtract()([helper2,hedgepf[l]])
                    pricekey[l] = helper2 
            price = Add()([price,priceincr])
            inputshelper = inputshelper + [stochvol]
            inputs = inputs + [BMincr]
        else:
            start = j
            break

inputs = inputs + inputshelper
pricekey = Concatenate()(pricekey)
localvol_trainlocvol = Model(inputs=inputs, outputs=pricekey)

# %%
#localvol_trainlocvol.summary()

# %% [markdown]
# Here we use a very small amount of trajectories due to the variance reduction coming from hedges just for purposes of illustration even though it already works relatively well. Below we consecutively train hedges or local volatilities.

# %%
Ltrain = 5*10**3

xtrain =([np.ones(Ltrain)] + [np.zeros(Ltrain) for key in keylist]+
         [np.random.normal(0,1,Ltrain) for i in range(N)]+
         [np.ones(Ltrain) for i in range(N)])

ytrain=np.zeros((Ltrain,len(P.keys())))
for i in range(Ltrain):
    for l in range(len(P.keys())):
        key = keylist[l]
        ytrain[i,l]= P[key]

# %% [markdown]
# In the sequel the actual training is performed:

# %%
import matplotlib.pyplot as plt

localvol_trainhedge.compile(optimizer='adam', 
              loss='mean_squared_error')
localvol_trainlocvol.compile(optimizer='adam', 
              loss='mean_squared_error')
for i in range(3):
    localvol_trainhedge.fit(x=xtrain,y=ytrain, epochs=15,verbose=True)
    x = localvol_trainhedge.get_weights()
    localvol_trainlocvol.set_weights(x)
    localvol_trainlocvol.fit(x=xtrain,y=ytrain, epochs=5,verbose=True)
    plt.hist(localvol_trainhedge.predict(xtrain)[:,0])
    plt.show()
    print(np.mean(localvol_trainhedge.predict(xtrain)[:,0]))
    y = localvol_trainlocvol.get_weights()
    localvol_trainhedge.set_weights(y)

# %% [markdown]
# Hedging helps to reduce variance tremendously, whence we are able to go for a classical means square calibration approach, which is implemented below with a custom loss function.

# %%
def custom_loss(y_true,y_pred):
    return K.mean((K.mean(y_pred,axis=0)-K.mean(y_true,axis=0))**2)

localvol_trainlocvol.compile(optimizer='adam', 
              loss=custom_loss)

# %%
for i in range(10):
    localvol_trainlocvol.fit(x=xtrain,y=ytrain, epochs=10, verbose=True,batch_size=10**3)
    plt.hist(localvol_trainlocvol.predict(xtrain)[:,:])
    plt.show()
    print(np.mean(localvol_trainlocvol.predict(xtrain)[:,:],axis=0))

# %%
Ltest = 10**6

xtest =([np.ones(Ltest)] + [np.zeros(Ltest) for key in keylist]+
         [np.random.normal(0,1,Ltest) for i in range(N)]+
         [np.ones(Ltest) for i in range(N)])

ytest=np.zeros((Ltest,len(P.keys())))
for i in range(Ltest):
    for l in range(len(P.keys())):
        key = keylist[l]
        ytest[i,l]= P[key]

# %%
plt.hist(localvol_trainlocvol.predict(xtest)[:,:])
plt.show()
print('Calibrated values:', np.mean(localvol_trainlocvol.predict(xtest)[:,:],axis=0))
print('Ground truth:', [P[key] for key in keylist])

# %%
P

# %% [markdown]
# ... not so bad.

# %%
#maturities = [0.1, 0.25, 0.5, 1.0]
#strikes = [0.8, 0.9, 1.0, 1.1, 1.2]
#for T in maturities:
#    for K in strikes:
#         P[(K,T)] = 1.0
#Lgen = 10**6
#keylist = P.keys()
#xgen =([np.ones(Lgen)] + [np.zeros(Lgen) for key in keylist]+
#       [np.random.normal(0,1,Lgen) for i in range(N)]+
#       [np.ones(Lgen) for i in range(N)])
#
#ygen=np.mean(localvol_trainlocvol.predict(xgen)[:,:],axis=0)
#for l in range(len(P.keys())):
#    key = keylist[l]
#    P[key] = ygen[1,l]


