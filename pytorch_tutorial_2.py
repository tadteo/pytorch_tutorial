#! /usr/bin/env python3

import numpy as np 

X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)

w = 0.0

#model prediction
def forward(x):
    return w*x


#loss = MSE
def loss(y,y_predicted):
    #MSE= 1/N  * (w)
    return((y_predicted-y)**2).mean()

#gradient
def gradient(x,y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')


learning_rate =0.01
n_iters=20

for epoch in range(n_iters):
    #predictio = forward pass
    y_predicted = forward(X)

    #loss
    l= loss(Y,y_predicted)

    #gradient
    dw= gradient(X,Y,y_predicted)

    #update weights
    w -= learning_rate*dw

    if epoch%1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.5f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')

###### with pytorch #####

import torch

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, requires_grad=True)

#model prediction
def forward(x):
    return w*x


#loss = MSE
def loss(y,y_predicted):
    #MSE= 1/N  * (w)
    return((y_predicted-y)**2).mean()

print(f'Prediction before training: f(5) = {forward(5):.3f}')


learning_rate =0.01
n_iters=100

for epoch in range(n_iters):
    #predictio = forward pass
    y_predicted = forward(X)

    #loss
    l= loss(Y,y_predicted)

    #gradient = backward
    l.backward() #dl/dw #it is not exactly as the numerical derivative

    #update weights
    with torch.no_grad():
        w -= learning_rate*w.grad

    #zero gradients for nbext epoch
    w.grad.zero_()

    if epoch%10 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.5f}')


print(f'Prediction after training: f(5) = {forward(5):.3f}')
