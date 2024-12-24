#CSC 4760 - Introduction to Deep learning
#May Wandyez gq5426
#Homework 7

#First thing's first, look through the code someone else wrote and ACTUALLY add comments so that it can be reverse engineered.
#The code can be found here (this is for you future self, you will forget where you got this from):
# " https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py   "

print("I am attempting to import the libraries")
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


############################################################################################
# Device configuration
#This is boilerplate for Pytorch, it can't work unless the device is set. 
#I WAS WRONG THIS IS SUPER IMPORTANT, THIS IS THE DEVICE NECESSARY FOR CREATING
#THE OBJECT OF YOUR MODEL CLASS
print("I am attempting to set up device configuration")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
###########################################################################################

#=-=-=-=-=-=-=-=-=-=-=CHANGE STUFF HERE, THESE ARE YOUR PAREMETERS=-=-=-=-=-=-=-=-=-=-=-=-
# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


#@~=-----------------------Get your dataset--------------------=~@
#This section appears to be loading data from the MNIST torchvision dataset.
#It downloads the data if you don't have it, otherwise it just looks for data to see
#if it is there. 
#It should be noted that it automatically transforms the dataset into a tensor
#so that it can be loaded into the model.

# MNIST dataset
#This is our data that we will be using to train the dataset
print("I am attempting to download the dataset")
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)
#This is the data that we will be using to TEST the dataset 
#I mean the variable name should be a giveaway, but if you need to read the variables
#to understand your code, then you don't have enough comments.
test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False, 
                                          transform=transforms.ToTensor())
#@~=------------------------------------------------------------=~@


#@~---------------------CREATE DATA LOADERS-------------------------~@
#This is the data loader section, you have tensors on hand, and
#now you need to actually shove them into the model. The end result is a 
#new object of the Dataloader type that will eventually be loaded into the model

#Please note that the dataloader includes a batch size feature
#this specifies the number of samples per batch

#Shuffle shuffles the data before loading it. Sounds like that could e
#useful to prevent overfitting

# Data loader
print("I am attempting to create the data loaders")
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
#@~------------------------------------------------------------~@

#============DEFINE THE MODEL (IMPORTANT!!!!!)======================
#This section is devoted to actually creating the model. Looks like
#the input for the class is the number of classes that you want to put
#and when it is created you have to specify the device it is running on.

# Convolutional neural network (two convolutional layers)
#Create the class (give it a name, inputs)
print("I am attempting to define the model")
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        #Looks like this makes use of the nn.Sequential function, which as far
        #as I can tell just allows creation of a single layer by stacking other layers
        #Let's take a look at the actual contents of the sequence
        #NN Conv2d
        #Looks like this is the convolutional layer we were so interested in
        #the kernel size is the size of the convolution (it's a 5x5 square)
        #out channels is the number of feature maps generated (in this case 16)
        #looks like there's only 1 in channel because it's grayscale
        #stride is stride, that's just how much the filter moves
        #padding is the zero padding in case you reach the border of the image
        
        #NN.BatchNorm2d() creates a 2d batch normalization depending on the number of inputs
        #because our previous output is 16, the input must be 16. Batch normaliazation is used
        #speed up training and reduce variance in data. Considering we just generated a fudge
        #ton of data through a convolution, this is a necessary step
        
        #NN.ReLu turns all values below zero to zero for your NN. In short it helps makes
        #sure neurons in your layers don't activate when they shouldn't. Don't worry about how
        #it works, just know that it makes the neurons work better.
        
        #The max pooling layer reduces the size of feature maps using a pooling operation
        #looks like the size of the pooling window iws 2x2 and the stride is 2. This doesn't
        #decrease the number of layers, but does reduce the feature map size within them
        #looks like when it's all said and done
        
        #looks like layer 2 is just the same as layer one but it has even more feature inputs (16 layers worth)
        
        #maybe I should add in an additional layer ya know? To show I vaguely know what I am doing and didn't
        #just copy the example
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        #using math weknow that the feature maps are now 7x7, and if you are confused in the future
        #and don't know what size your feature maps are, just run the program, print a feature map out
        #and count manually - it's probably faster than doing the math in your head or on paper.
        #there are 32 layers now from layer 2, The number of classes is based on the number of classes specified
        #by the user in the hyper parameters (this is your output, the thing you will be checking against in your
        #classification section)
        self.fc = nn.Linear(7*7*32, num_classes)
    
    
    #This is the forward function for the neural network, it is used to create outputs from the model
    def forward(self, x):
        #output is based on layer one for whatever is fed into the model
        out = self.layer1(x)
        #output for the second layer is based on the output for the first layer
        out = self.layer2(out)
        #not sure why the batch size is zero, but the reshaping is automatically based on however
        #out is currently shaped
        out = out.reshape(out.size(0), -1)
        #out is fed into the fully connected layer (this is just a single FC layer for a NN)
        out = self.fc(out)
        #the output is returned (What does the output actually look like? I know it is returned? Is it the class for the 
        # particular image fed in? Surely it must be)
        return out

#===================================================================
model = ConvNet(num_classes).to(device)



############################LOSS FUNCTION DEFINITION###################################
# Loss and optimizer
print("I am attempting to define the loss")
#this is your loss function, it is based on softmax and some other stuff. Don't worry about it.
#you're not good enough at math to grasp the true potential. Just know what you can slap
#whatever loss function you want in this section here
criterion = nn.CrossEntropyLoss()
############################################################################


#$$$$$$$$$$$$$$$$$$$$$$$$ OPTIMIZER DEFINITION $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#This is your optimizer, it updates the weights and biases of your neural
#network during training. Remember, your model is named MODEL (honestly
# that confuses me, it looks like its referencing a variable if it is
# so simply named. I prefer long complicated variable names)

#This optimizer is an off the shelf model called "Adam", it takes the parameters
#from the model and the learning rate (thing that determines step size)
#Don't worry about it, there's different optimization functions that
#have different strengths, just know that this is where your learning rate matters
#and affects how well your model will be trained. Actually now that I type
#that out I should be REALLY worried about it. Huh.

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#**********************TRAINING SECTION*************************************
#Congrats, you defined a model. It's just a bunch of random numbers flowing 
#through a design, but using your loss function to see how wrong you are, and your
#optimizer to see how much you should correct, you can tweak all the tiny 
#numbers inside the model to create outputs you want.

# Train the model
print("I am attempting to train the model")
#The number of steps we need to make is EQUAL in length to the amount of data we feed in (duh)
total_step = len(train_loader)

#epochs are how many times you are going to run through the data. Check your parameters
#if your error isn't getting small enough, increase the number of epochs, or mess around
#with the learning rate. At a certain point your program is too complicated and you have
#to poke it with a stick to make it do what you want.

for epoch in range(num_epochs):
    #you know I've never see the enumerate function before, I think it's just a loop
    #the point is you are getting your image label pairings and counting i how many times
    #you've actually done it.
    for i, (images, labels) in enumerate(train_loader):
        #move the actual data you are considering to the place where the funny calculations
        #are happening. 
        images = images.to(device)
        labels = labels.to(device)
        
        
        # Forward pass
        #create outputs from the inputs using the model, check out wrong they are using the loss
        #function
        outputs = model(images) #creating outputs from the inputs (training data)
        loss = criterion(outputs, labels) #calculating the loss.
        
        # Backward and optimize
        #Okay admittedly I forgot what these are off the top of my head. Time to google them.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #print how we are doing every 100th data piece
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


#*****************************************************************************
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Test the model
#Congrats, you built your model, you've messed with its internal parameters during training, and 

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Save the model checkpoint
#Congrats, you finished your model, time to save it so you can use it later! You will have to load
#the model separately when you import it though (I've had issues with this)
torch.save(model.state_dict(), 'model.ckpt')


#Great, we labeled the code to make sense of it, wouldn't it be fun if we made it run slower by adding in more layers? (Remember, the layers have to actually add more layers)
#to the neural network architecture, OR instead 