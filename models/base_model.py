
import torch
import torch.nn as nn
import inspect
import json
import os
import numpy as np
import sys
sys.path.append("../")
from BERTLocRNA.models.attention import Attention_mask
from BERTLocRNA.models.layers import *
from torch import Tensor

#make the code reproducible 
# Set random seed for NumPy
np.random.seed(42)

# Set random seed for PyTorch
torch.manual_seed(42)

# If using GPU, set random seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
path_join = lambda *path: os.path.abspath(os.path.join(*path))

device = torch.device("cuda" if torch.cuda.is_available else "cpu")



class CustomizedModel(nn.Module):
    '''
    The model run for all embeddings
    '''
    def __init__(self, config):
                                                                                          
        super(CustomizedModel, self).__init__()
        self.config = config
        neurons = int(self.config.headnum*self.config.hidden_dim)
        #attention layers
        #fully connected layers
        self.fc1 = nn.Linear(neurons, self.config.fc_dim).to(device)
        self.fc2 = nn.Linear(self.config.fc_dim + self.config.RNA_dim, self.config.nb_classes)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(self.config.drop_flat)
        self.att_weight = None
        self.headnum = self.config.headnum
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.config.RNA_order),embedding_dim=self.config.RNA_dim,_weight=torch.zeros((len(self.config.RNA_order), self.config.RNA_dim)))
        self.sigmoid = nn.Sigmoid()
        
        self.Actvation = Actvation(self.config.activation)
        self.Attention_layer = Attention_mask(hidden = self.config.hidden_dim, att_dim = self.config.dim_attention, 
                                            r = self.config.headnum, activation = self.config.activation_att, 
                                            return_attention = True, attention_regularizer_weight = self.config.Att_regularizer_weight, 
                                            normalize = self.config.normalizeatt,attmod = self.config.attmod,
                                            sharp_beta = self.config.sharp_beta)
        #using pooling layer to downsample the sequence
        self.maxpool = nn.MaxPool1d(self.config.pooling_size, stride = self.config.pooling_size)

        
        

    def print_init_parameters(self):
        init_params = inspect.signature(self.__init__).parameters
        param_names = [param for param in init_params if param != 'self']
        for param_name in param_names:
            param_value = getattr(self, param_name)
            print(f"{param_name}: {param_value}")

    def Att(self, embed, x_mask):   

        embed_output = embed*x_mask 
        embed_output = torch.cat((embed_output,x_mask), dim = 1)
        att1,att1_A = self.Attention_layer(embed_output, masks = True)
        
        self.att_weight = att1_A
        att1 = att1.transpose(1,2)
        
        return att1

    def mergelncRNA(self, RNA_type):
        idx = torch.where(RNA_type == self.config.RNA_order.index("lincRNA"))[0]
        if len(idx) > 0:
            RNA_type[idx] = self.config.RNA_order.index("lncRNA")
        return RNA_type
    
    def forward(self, embed : Tensor, x_mask : Tensor, RNA_type : Tensor):

        #The input should be embedding of different pre-trained methods
        RNA_type = RNA_type.long()
        embed = self.dropout(self.maxpool(embed))
        output = self.Att(embed, x_mask) #[hidden, heads] 
        output = self.flatten(output)
        
        #getting RNA types identify layer
        embedding_output = self.embedding_layer(self.mergelncRNA(RNA_type))#n*4
        output = self.fc1(output)
        output = torch.cat((output, embedding_output), dim=1)
        output = self.Actvation(output)
        output = self.dropout(output)
        output = self.fc2(output)
        pred = self.sigmoid(output)

        return pred
        # return output

