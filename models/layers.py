import torch.nn as nn
import torch
import math

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)



class Pooling(nn.Module):
    def __init__(self, type, pooling_size):
        super(Pooling, self).__init__()
        self.type = type
        self.maxpool = nn.MaxPool1d(pooling_size, stride = pooling_size)
        self.meanpool = nn.AvgPool1d(pooling_size, stride = pooling_size)
        if self.type == "None":
            self.layer_name = "NoPooling"
        else:
            self.layer_name = f"{self.type}_pooling_{pooling_size}"
    def forward(self, x):
        if self.type == "max":
            x = self.maxpool(x)
        elif self.type == "mean":
            x = self.meanpool(x)
        elif self.type == "None":
            pass
        return x
    

    
class Activation(nn.Module):
    def __init__(self, name):
        super(Activation, self).__init__()
        self.name = name
        self.layer_name = None

    def forward(self, x):
        if self.name == "relu":
            x = torch.nn.functional.relu(x)
            self.layer_name = "Activation_ReLU"
        elif self.name == "gelu":
            x = torch.nn.functional.gelu(x)
            self.layer_name = "Activation_GeLU"
        elif self.name == "leaky":
            x = torch.nn.functional.leaky_relu(x)
            self.layer_name = "Activation_Leaky"

        return x
    def __repr__(self):
        return f"{self.name}()"
        
    

