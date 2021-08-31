#! /usr/bin/env python3

import torch
import numpy as np

#pytorch is based on tensors

x = torch.empty(2,2)  #create a 2x2 matrix
x = torch.random(2,2) #the same but with random numbers
x = torch.ones(2,2) #the same but with random numbers


x = torch. random(2,2, dtype=torch.float16) #normally pytorch use float 32 but we can change using dtype
print(x, x.dtype)

x = torch.random(2,2)
y = torch.random(2,2)

#thiis two are equivalent
z = x+y
z = torch.add(x,y)

z= x-y
z = torch.sub(x,y)


x.add_(y) #normally ion pytorch the operation wich end with underscore perform inplace operations


x = torch.random(5,3)

#slicing as in python
print( x[1,:])

print( x[2,3].item()) #to have the single item otherwise a tensor with one element is returned

#resize tensor with the .view() function

x = x.view(15) #unidimensional
x = x.view(-1,5) #-1 automatically determine the other value

#to convert to a numpy array

a = torch.ones(2,5)
print(a)
b = a.numpy()
print(b)
c = torch.from_numpy(b) 

#if in cpu both are a and b are pointing to the same memory space


#You can choose the device CPU or GPU
if torch.cuda.is_available(): #to check if cuda is available
    device = torch.device("cuda") #if cuda is avaialable
    x= torch.ones(5, device=device) #in gpu
    y= torch.ones(5) #in cpu
    y= y.to(device)
    z= x+y #now I can sum them up 
    #however numpy works only on cpu so
    z.to("cpu")
    z = z.numpy()


#If you know that you need to calculate the gradient to this tensor add requires_grad=True for optimization purpuses (like nn layers_)

x = torch.ones(5,2,requires_grad=True)
print(x)


########################################## ######################################### ############################################

x = torch.ones(5,2,requires_grad=True) #it will create a computational graph
print(x)

y = x+2 
print(y)

z= y*y*2
z = z.mean()
print(z)

z.backward() #dz/dx #to execute the gradient # if we don't specify reqiuires_grad=True this will create an error
print(x.grad)

### In case z is nto scalar pay attention to the math., since we will need to pass to the backward function a vector parameter

x = torch.ones(5,2,requires_grad=True) #it will create a computational graph
print(x)

y = x+2 
print(y)

z= y*y*2
print(z)

v= torch.tensor([0.1,0.1,0.001], dtype=torch.float32)

z.backward(v) #dz/dx #this will produce a vector jacobian product
print(x.grad)


#If we don't want to add a specific operation to the computation graph
# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():
#   y = x+2
#   print(y)



# The backward() function will sum up the gradient so in a training when we change epocch we need to recall to format the gradient to do that

weights = torch.ones(5, requires_grad=True)
z.grad.zero_()


