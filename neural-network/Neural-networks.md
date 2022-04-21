# Neural Networks using Pytorch

Pytorch is a framework of different machine learning algorithms that support CUDA parallelization. Pytorch have a data structure similar to Numpy arrays that are called Tensors.

This is a simple implementation of a neural network that learn from [MNIST dataset](http://yann.lecun.com/exdb/mnist/) using a single layer model.

The model use [Pytorch 1.11](https://pytorch.org/)


```python
import torch
import torch.nn as nn # neural networks
import torch.optim as optim # optimization
import torch.nn.functional as Tfunc 
from torch.utils.data import DataLoader 
import torchvision.datasets as Tdatasets 
import torchvision.transforms as Ttransforms
import matplotlib.pyplot as plt
import numpy as np
```

Let's set up a couple of parameters and load our dataset. The dataset is MNIST and each image is going to be transformed to Pytorch tensor. The loader is going to take a batch of 64 images that are shuffled each time we call it.


```python
#this is the size of our batch of data
batch_size = 64

#our train dataset from MNIST downloaded into dataset folder if true and transformed into a Pytorch tensor
train_dataset = Tdatasets.MNIST( root="dataset/", train=True, transform=Ttransforms.ToTensor(), download=True )

#we use the dataloader to load the train dataset of bach size and is going to be shuffled
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
```

For more information about neural networks visit [this awesome example](https://aegeorge42.github.io/)

Let's build our Neural Network model. We are going to use: 

- One input layer

- One hidden layer of 50 nodes

- One output layer

- All neurons are fully connected

- [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))(Rectified Linear Unit) activation function

- Backpropagation is automatically done by Pytorch using autograd

Let's define the class with our model:


```python
class NN(nn.Module):
    #initialization function
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # one input layer 
        self.fc1 = nn.Linear(input_size, 50)
        # one hidden layer
        self.fc2 = nn.Linear(50, num_classes)
        
    #forward function
    def forward(self, x):
        # one output layer
        #use ReLU as activation function
        x = Tfunc.relu(self.fc1(x))
        x = self.fc2(x)
        return x 
```

### Training and parameters


```python
#use cuda if available for parallelization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#amount of pixels of each image
input_size = 784 # 28x28 = 784, size of MNIST images (grayscale)

#number of clases 
num_classes = 10 #there are 10 digits from 0 to 9

#we set to this value for this example
learning_rate = 0.001

#number of times (epochs) that is going to perform the training  
num_epochs = 4
```


```python
device
```




    device(type='cuda')



We have a NVIDIA GPU so is going to use it for the model, this is the good thing of been a **Gamer** :)


```python
#model defined previously in our NN class and is sent to the cuda device in this case
model = NN(input_size=input_size, num_classes=num_classes).to(device)

#We are using a Cross entropy criteria for our loss
criterion = nn.CrossEntropyLoss()

#The optimizer is Adam with the learning rate that we previously set
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```

### Training loop


```python
for epoch in range(num_epochs):
    print(f"Epoch: {epoch}")
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Get to correct shape, 28x28->784
        # -1 will flatten all outer dimensions into one
        data = data.reshape(data.shape[0], -1) 

        # forward propagation
        scores = model(data)
        loss = criterion(scores, targets)

        # zero previous gradients
        optimizer.zero_grad()
        
        # back-propagation
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
```

    Epoch: 0
    Epoch: 1
    Epoch: 2
    Epoch: 3


### Testing


```python
test_dataset = Tdatasets.MNIST(root="dataset/", train=False, transform=Ttransforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
```


```python
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy"
            f" {float(num_correct) / float(num_samples) * 100:.2f}"
        )

    model.train()
```


```python
check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
```

    Got 58145 / 60000 with accuracy 96.91
    Got 9646 / 10000 with accuracy 96.46

