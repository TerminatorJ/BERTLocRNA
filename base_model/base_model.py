from hier_attention_mask_torch import Attention_mask
import torch
import torch.nn as nn
import pytorch_lightning as pl
import time
import inspect
import re
import json
import os
import numpy as np
import sys
sys.path.append("../")
from utils import *
from layers import *

#make the code reproducible 
# Set random seed for NumPy
np.random.seed(42)

# Set random seed for PyTorch
torch.manual_seed(42)

# If using GPU, set random seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

parent_directory = os.path.dirname(os.getcwd())
with open(path_join(parent_directory, "vocabulary.json"), 'r') as json_file:
    vocab = json.load(json_file)

class CustomizedModel(nn.Module):
    '''
    The model run for all embeddings
    '''
    def __init__(self, config):
                                                                                          
        super(CustomizedModel, self).__init__()

        #fully connected layers
        
        self.fc2 = nn.Linear(config.fc_dim + config.RNA_dim, config.nb_classes)
        
        #embedding layer to expand the input dimention

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(config.drop_flat)
        self.att = None
        self.embedding_layer = nn.Embedding(num_embeddings=len(config.RNA_order),embedding_dim=config.RNA_dim,_weight=torch.zeros((len(config.RNA_order), config.RNA_dim)))
        self.sigmoid = nn.Sigmoid()
        
        self.Actvation = Actvation(config.activation)
        
        self.FC_block = nn.Sequential(self.fc1,
                                        self.Actvation(config.activation),
                                        self.dropout,
                                        self.fc2,
                                        self.sigmoid)
        

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
        
        self.att = att1_A
        att1 = att1.transpose(1,2)
        
        return att1

    
    def forward(self, embed, x_mask, RNA_type):
        #neuron number should be define until the run time
        neurons = int(self.config.headnum*embed.shape[1])
        self.fc1 = nn.Linear(neurons, self.config.fc_dim)
        #attention layers
        self.Attention_layer = Attention_mask(hidden = embed.shape[1], att_dim = self.config.dim_attention, 
                                        r = self.config.headnum, activation = self.config.activation_att, 
                                        return_attention = True, attention_regularizer_weight = self.config.Att_regularizer_weight, 
                                        normalize = self.config.normalizeatt,attmod = self.config.attmod,
                                        sharp_beta = self.config.sharp_beta)

        #The input should be embedding of different pre-trained methods
        #expand the mask to 3d
        x_mask = x_mask.unsqueeze(1).float()  
        output = self.Att(embed, x_mask) #[hidden, heads] 
        output = self.flatten(output)
        
        #getting RNA types identify layer
        embedding_output = self.embedding_layer(RNA_type)#n*4
        output = self.fc1(output)
        output = torch.cat((output, embedding_output), dim=1)
        output = self.Actvation(output)
        output = self.dropout(output)
        output = self.fc2(output)
        pred = self.sigmoid(output)

        return pred


class LightningModel(pl.LightningModule):
    def __init__(self, config, class_weights = None):
        super(LightningModel, self).__init__()
        #only used to extract the RNA types
        self.network = CustomizedModel(config).to(device)
        self.class_weights = class_weights
        self.epoch_start_time = None
        self.loss_fn = nn.BCELoss()

    def naive_loss(self, y_pred, y_true, ohem=False, focal=False):

        loss_weight_ = self.class_weights
        loss_weight = []
        for i in range(self.nb_classes):
        # initialize weights
            loss_weight.append(torch.tensor(loss_weight_[i],requires_grad=False, device=device))
        num_task = self.nb_classes
        num_examples = y_true.shape[0]

        def binary_cross_entropy(x, y,focal=True):
            alpha = 0.75
            gamma = 2

            pt = x * y + (1 - x) * (1 - y)
            at = alpha * y + (1 - alpha)* (1 - y)

            # focal loss
            if focal:
                loss = -at*(1-pt)**(gamma)*(torch.log(x) * y + torch.log(1 - x) * (1 - y))
            else:
                epsilon = 1e-4  # Small epsilon value
                # Add epsilon to x to prevent taking the logarithm of 0
                x = torch.clamp(x, epsilon, 1 - epsilon)
                loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
            return loss
        loss_output = torch.zeros(num_examples).to(device = device)
        for i in range(num_task):
            if loss_weight != None:
                out = loss_weight[i]*binary_cross_entropy(y_pred[:,i],y_true[:,i],focal)                
                loss_output += out
            else:
                loss_output += binary_cross_entropy(y_pred[:, i],y_true[:,i],focal)

        # Online Hard Example Mining
        if ohem:
            val, idx = torch.topk(loss_output,int(0.7*num_examples))
            loss_output[loss_output<val[-1]] = 0
        loss = torch.sum(loss_output)/num_examples
        return loss
    def binary_accuracy(self, y_pred, y_true):
        # Round the predicted values to 0 or 1
        y_pred_rounded = torch.round(y_pred)
        # Calculate the number of correct predictions
        correct = (y_pred_rounded == y_true).float().sum()
        # Calculate the accuracy
        accuracy = correct / y_true.numel()
        return accuracy
    
    def categorical_accuracy(self, y_pred, y_true):
        # Get the index of the maximum value (predicted class) along the second dimension
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        # Compare the predicted class with the target class and calculate the mean accuracy
        return (y_pred == y_true).float().mean()

    def forward(self, embed, mask, RNA_type = None):
        x = x.to(device)
        mask = mask.to(device)
        if RNA_type != None:
            RNA_type = RNA_type.to(device)
        pred = self.network(embed, mask, RNA_type)
        return pred
    
    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer_cls"])(self.parameters(), lr = self.config["lr"], weight_decay = self.config["weight_decay"])
        return optimizer

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch duration: {epoch_time:.2f} seconds")
    def _attention_regularizer(self, attention):
        batch_size = attention.shape[0]
        headnum = self.network.headnum
        identity = torch.eye(headnum).to(device)  # [r,r]
        temp = torch.bmm(attention, attention.transpose(1, 2)) - identity  # [none, r, r]
        penal = 0.001 * torch.sum(temp**2) / batch_size
        return penal

    def training_step(self, batch, **kwargs):

        x, mask, RNA_type, y = batch
        y = y.to(torch.float32)
        y_pred = self.forward(x, mask, RNA_type)

        if self.class_weights is None:
            loss = self.loss_fn(y_pred, y)
        else:
            loss = self.naive_loss(y_pred, y)

        #Using the gradient clip to protect from gradient exploration
        if self.config["gradient_clip"]:
            nn.utils.clip_grad_norm_(self.network.parameters(), 1)

        l1_regularization = torch.tensor(0., device=device)
        for name, param in self.network.named_parameters(): 
            if 'Attention_layer.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention_layer.W2' in name:
                l1_regularization += torch.norm(param, p=1)

        loss += l1_regularization*0.001

        loss += self._attention_regularizer(torch.transpose(self.network.att, 1, 2))
        self.log("train_loss", loss, on_epoch = True, on_step = True)

        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)
        
        self.log('train categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('train categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('train binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)
 
        return loss
    def categorical_accuracy_strict(self, y_pred, y_true):
    # Find the index of the maximum value in each row (i.e., the predicted class)
        y_pred_class = torch.round(y_pred)
        com = y_pred_class == y_true
        correct = com.all(dim=1).sum()
        sample_num = y_true.size(0)
        accuracy = correct / sample_num
        return accuracy
    def validation_step(self, batch):

        x, mask, RNA_type, y= batch
        y = y.to(torch.float32)
        y_pred = self.forward(x, mask, RNA_type)

        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)

        self.log('val categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('val categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('val binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)

        if self.class_weights is None:
            loss = self.loss_fn(y_pred, y)
        else:
            loss = self.naive_loss(y_pred, y)
        l1_regularization = torch.tensor(0., device=device)
        for name, param in self.network.named_parameters(): 
            if 'Attention_layer.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention_layer.W2' in name:
                l1_regularization += torch.norm(param, p=1)

        loss += l1_regularization*0.001
        loss += self._attention_regularizer(torch.transpose(self.network.att, 1, 2))
           

        self.log("val_loss", loss, on_epoch = True, on_step = True)

        return {"categorical_accuracy": categorical_accuracy, "categorical_accuracy_strict":categorical_accuracy_strict,
                "binary_accuracy": binary_accuracy}
    def print_init_parameters(self):
        init_params = inspect.signature(self.__init__).parameters
        param_names = [param for param in init_params if param != 'self']
        for param_name in param_names:
            param_value = getattr(self, param_name)
            print(f"{param_name}: {param_value}")



