#MAY WANDYEZ GQ5426 CSC 4760 HOMEWORK 3
#Objectives:
"""
Load Homework3.mat in python, use
weight and horsepower to describe each
sample

Clean the training data set to remove
NaN.

Train a linear regression model to predict
the attribute of acceleration.

Plot the samples and plane representing the 
trained regression model in one figure

Upload the plot and code in one file.
"""

#STEP 1: Import a .mat file into python.
#=-=-=-=-==-=-=-=-=-=-=--=-=-=-=-=--=-=-=-=
import scipy.io
#Turns the .mat file into a dictionary
mat = scipy.io.loadmat('Homework3.mat')

#Turns the rows of the dictionary into variables we can actually use.
Hp = mat['Horsepower']
We = mat['Weight']
Ac = mat['Acceleration']
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


#STEP 2:Train a linear regression model
#to predict the attribute of acceleration
#Based on weight and hp
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=
import torch
from torch import nn
from torch.autograd import variable
import numpy as np


#clean the data
clean_indices = np.logical_not(np.isnan(Hp) | np.isnan(We) | np.isnan(Ac))
Hp = Hp[clean_indices]
We = We[clean_indices]
Ac = Ac[clean_indices]


#turn the two arrays into one array
x_train = np.column_stack((Hp, We))
z_train = Ac.reshape(-1, 1)

#Troubleshooting the data to see why the regression isn't working
#print(x_train)
#print(z_train)

#turn the arrays into PyTorch tensors 
x_train = torch.from_numpy(x_train.astype(np.float32))
z_train = torch.from_numpy(z_train.astype(np.float32))

print(x_train.shape)
print(x_train.dtype)
print(z_train.shape)
print(z_train.dtype)


#Use pytorch to create a linear regression model
class linearRegression(nn.Module):
    def __init__(self):
        super(linearRegression,self).__init__()
        #Define a linear layer with 2 inputs and ONE output
        self.linear = nn.Linear(2,1)
    
    def forward(self,x):
        out = self.linear(x)
        return out

#create an instance of the regression model to use.
model = linearRegression()

#Create the loss function and optimizer
#If something is going wrong in the loss function
#it is probably happening here.
criterion = nn.MSELoss()
#it was the rate of loss, it was too high, I reduced it A LOT
optimizer = torch.optim.SGD(model.parameters(),lr=1e-7)

#actual training for the existing model
#number of times to train the model
num_epochs = 1000
#define inputs and target
inputs = x_train
target = z_train

for epoch in range(num_epochs):

    #Compute predicted output based upon input
    out = model(inputs)
    #compute the loss between the output of the model and target
    loss = criterion(out,target)
    
    #update weights based on backwards pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print the loss on every loop to see why it keeps telling me there is nan
    #for the loss function
    #Neat, looks like the loss function was working but then tended towards
    #INFINITY
    print(loss)
    
    #print loss every 20th pass. (Maybe this is where the error is)
    if (epoch+1)%20 == 0:
        print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')

#Change the model's mode to evaluation
model.eval()

#Generate predicted data set using the initial data set
with torch.no_grad():
    predict = model(x_train).detach().numpy()
    



#STEP 3: PLOT THE DATA
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')

# Plot original data
ax.scatter(Hp, We, Ac, c='r', marker='o', label='Original Data')

# Plot the surface based on X, Y, and predict_2d
#ax.plot_surface(Hp, We, predict, alpha=0.5, cmap='viridis', label='Fitted Surface')

#I plotted a plane, seemed like it was in the right place
X, Y = np.meshgrid(np.linspace(min(Hp), max(Hp), 100),
                   np.linspace(min(We), max(We), 100))

#Predict wasn't working, but this line of code seems to have acomplished the same thing. Yay!
Z = model(torch.Tensor(np.column_stack((X.flatten(), Y.flatten())))).detach().numpy().reshape(X.shape)
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', label='Fitted Surface')


ax.set_xlabel('Horsepower')
ax.set_ylabel('Weight')
ax.set_zlabel('Acceleration')

plt.legend()
plt.title('Linear Regression Fit')

plt.show()

# Plot fitted line
#ax.plot(Hp.flatten(), We.flatten(), predict.flatten(), c='b', label='Fitted Line')
"""
#I plotted a plane, seemed like it was in the right place
X, Y = np.meshgrid(np.linspace(min(Hp), max(Hp), 100),
                   np.linspace(min(We), max(We), 100))
Z = model(torch.Tensor(np.column_stack((X.flatten(), Y.flatten())))).detach().numpy().reshape(X.shape)
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', label='Fitted Surface')
"""


