
import torch.nn as nn

## Define the NN architecture
## NN with 3 hidden layer, s=[11, 256, 256] -> 8*[256,1]

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # The first hidden layer has 256 neurons
        self.fc1 = nn.Linear(11, 256)
        # The second hidden layer has 256 neurons
        self.fc2 = nn.Linear(256, 256)
        
        # The final layer has 8 heads and 1 output neuron
        # self.fc3 = nn.Linear(256, 256)
        # self.fc4 = nn.Linear(256, 256)
        # self.fc5 = nn.Linear(256, 256)
        # self.fc6 = nn.Linear(256, 256)
        # self.fc7 = nn.Linear(256, 256)
        # self.fc8 = nn.Linear(256, 256)
        # self.fc9 = nn.Linear(256, 256)
        # self.fc10 = nn.Linear(256, 256)
        
        

        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.1)

        # Mish activation
        self.mish = nn.Mish()


    def forward(self, x):
        # add first fully connected layer
        x = self.fc1(x)
        x = self.mish(x)
        x = self.dropout(x)
        # add second fully connected layer
        x = self.fc2(x)
        x = self.mish(x)
        output = self.dropout(x)
        return output
    
