#CSC 4760 Homework 6 May Wandyez - gq5426
#The objective of this homework assignment is to design a feed forward
#neural network for the purpose of handwritten digits classification
#Looks like the default model loads images from some default library
#Which means since it already works, all that has to be done
#is move the model to some other library.

print("I am attempting to import the libraries!")
#Import relevant libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#It should be noted that I am not sure if these libraries will actually work.
print("I have loaded the libraries, onto device configuration!")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("I have configured the device for cuda or CPU!")

# Hyper-parameters 
#input size is the total number of INPUT neurons
#hidden side is the total number of HIDDEN neurons
#num_classes is the total number of OUTPUT neurons
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001
print("I have created the variables for input parameters")


#OKAY, here is the dataset we need to train for handwriting recognition.
#We need to try to point this to the right place so we can load HANDWRITING instead.
print("I am downloading the data set!")
# MNIST dataset 
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

print("Data set downloaded! I am now loading the dataset!")
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

print("The data has been loaded, onto defining the model!")
# Fully connected neural network with one hidden layer
#Okay how the fudge does this work, time to figure it out!
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

print("The model has been defined, time to train the model!")
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #For every 100th epoch, tell us the current error
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

print("The model has been trained, onto TESTING the model")
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')