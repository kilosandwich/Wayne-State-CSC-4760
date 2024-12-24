#Gq5426 CSC 4760 Homework 8
#Instructions:
"""
*************
INSTRUCTIONS:
*************
Train a Recurrent Neural Network for training handwritten digits classification
A sample model has been provided here:

https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/recurrent_neural_network/main.py

"""
"""
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
Solution Strategy:
&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

A Recurrent Neural Network feeds information to itself in the same layer 
alongside simply feeding forward information.

The plan is to review how the sample model is built, then to edit its structure
to try to improve it - thus demonstrating knowledge of Recurrent Neural Networks

The first step is to go through the sample model, and comment it more extensively

"""


#################IMPORT LIBRARIES##################
print("Importing libraries")
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
###################################################


#$$$$$$$$$$$$$DEVICE CONFIGURATION$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# Device configuration
#We've talked about this in prior labs, this is designating where all the
#calculations are going to take place. It's boilerplate and inescapable.
#I'm not sure why you would use something other than CUDA, I guess if you
#have a really big CPU?
print("Configuring device")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

#===================================================================
#=-=-=-=-=-=--= PARAMETERS - CHANGE THESE=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Hyper-parameters
#THis is the parameters for the model, most of this looks the same, but
#It's not QUITE the same. We will have to get back to this later to make
#sure it does what we think it does.
print("Defining parameters")
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01
#===================================================================


#0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-
#---------------Data Aquisition and Dataloading------------------------
# MNIST dataset
#This is the dataset for handwritten digit classification, it has to be downlaoded
#if you have not downloaded it already. The data gets automatically transformed into a tensor
#so that it can be fed into the dataloader.
print("Downloading data")
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader
#You've retrieved your data, now change it into a dataloader object to be
#fed into the neural network.
#Interestingly, the batches appear to be shuffled to improve training -
#but not shuffled for the test dataloader
print("Creating dataloaders")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
#-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0



####################################################################################
#======================DEFINING THE MODEL===========================================
#Considering every single section of this model has otherwise been identical to previous
#sample models, it's pretty obvious this is where all the changes are going on.
#but what the heck are the changes? How can we improve them? Time to read this nonsense and try
#to understand what each and every command does
print("Defining the RNN class")



# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    #This is the initialization function for the RNN
    #The input size is changeable based on hyper parameters
    #the hidden size refers to the number of neurons in the hidden layers
    #the number of layers refers to the number of layers for each of the hidden layers
    #the number of classes refers to the classification problem being solved.
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        #define internal variable for size of hidden layer based upon user input
        self.hidden_size = hidden_size
        #define internal variable for number of layers based upon used input
        self.num_layers = num_layers
        
        self.bigN = 1
        print("Let's make the hidden layers ", self.bigN, " timeslarger!")
        
        #Looks like this is a Long Short Term Memory layer. It connects from the input
        #and outputs to a hidden layer.
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #Let's try adding a new layer, that sounds like a fun challenge.
        #Now I COULD have just increased the number of layers on the initial layer, but where is the fun in that?
        #I thought it would be more interesting to create a brand new layer that has a different internal size.
        #You know what would be funny? If i initialized a random number for the multiple. CHAOS!!!
        #self.lstm2 = nn.LSTM(hidden_size, hidden_size*self.bigN, num_layers, batch_first=True)
        #this layer connects from the hidden size to the number of classes. This is the
        #Output layer
        self.fc = nn.Linear(hidden_size*self.bigN, num_classes)
        
        self.workOnce = 0

    
    #This is the feedforward function. This looks WAY different
    #x represents the input tensor to be fed into the RNN
    def forward(self, x):

        #initialize inputs, we will recycle these later
        if self.workOnce == 0:
            self.h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            self.c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            self.workOnce = 1;
        
        #FORWARD STEP
        #The first step is to feed the inputs (represented by x) into the LSTM, which has been initialized
        #with zeroes
        # Forward propagate LSTM
        self.h0 = self.h0.detach()
        self.c0 = self.c0.detach()
        out, (self.h0, self.c0)= self.lstm(x, (self.h0.detach(), self.c0.detach()))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        
        out = self.fc(out[:, -1, :])
        return out
#####################################################################################

#DEFINE THE MODEL
print("Creating RNN instance")
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)




#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#-------------------TRAINING SECTION------------------------------------------
#This section appeals to be identical to previous data section
# Loss and optimizer
#LOSS FUNCTION - How wrong are we?
print("Defining the loss function")
criterion = nn.CrossEntropyLoss()
#OPTIMIZER FUNCTION - How do we correct knowing how wrong we are?
print("Defining the optimization function")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
print("Training the model")
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
#-------------------------------------------------------------------------------
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# Test the model
print("Testing the model")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')