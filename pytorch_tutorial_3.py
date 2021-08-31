#! /usr/bin/env python3

# Now we will use the pytorch loss and pytorch optimizer

###### with pytorch #####

# 1) Design model (input, output size, forward pass)
# 2) Construct the loss and optimizer
# 3) Training loop:
#       - forward pass: compute prediction
#       - backward pass: calculate gradients
#       - update weights

import torch
import torch.nn as nn


X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)


X_test= torch.tensor([5],dtype=torch.float32)

n_saples, n_features = X.shape
input_size = n_features
output_size = n_features

# model = nn.Linear(input_size,output_size)

class LinearRegression(nn.Module):

    def __init__(self,input_dim, output_dim):
        super(LinearRegression,self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim,output_dim)

    def forward(self,x):
        return self.lin(x)

model = LinearRegression(input_size,output_size)


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')


learning_rate =0.01
n_iters=100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(n_iters):
    #predictio = forward pass
    y_predicted = model(X)

    #loss
    l= loss(Y,y_predicted)

    #gradient = backward
    l.backward() #dl/dw #it is not exactly as the numerical derivative

    #update weights
    optimizer.step()

    #zero gradients for nbext epoch
    optimizer.zero_grad()

    if epoch%10 == 0:
        [w,b]=model.parameters()
        print(f'epoch {epoch}: w = {w[0][0].item():.3f}, loss = {l:.5f}')


print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')
