##This is an implementation of fine-tuning PLM using LoRA adaption
##19/2/2024


from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, DataCollatorWithPadding, AutoModel
import os
import torch
from torch import nn
from typing import List, Tuple, Dict, Union, Sequence
from torch import Tensor
from BERTLocRNA.models.layers import *



class LocalizationHead(nn.Module):

    def __init__(self, model_config : Dict = None, length = 1):
        super(LocalizationHead, self).__init__()
        self.length = length
        self.model_config = model_config
        self.maxpool = nn.MaxPool1d(model_config.pooling_size, stride = model_config.pooling_size)
        self.embedding_layer = nn.Embedding(num_embeddings=len(model_config.RNA_order),embedding_dim=model_config.RNA_dim,_weight=torch.zeros((len(model_config.RNA_order), model_config.RNA_dim)))
        self.activation = Activation(model_config.activation)
        self.flat_dim = int(model_config.hidden_dim*self.length + model_config.RNA_dim)
        self.last_layer = nn.Linear(self.flat_dim , model_config.nb_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(model_config.drop_flat)
    def mergelncRNA(self, RNA_type):
        idx = torch.where(RNA_type == self.model_config.RNA_order.index("lincRNA"))[0]
        if len(idx) > 0:
            RNA_type[idx] = self.model_config.RNA_order.index("lncRNA")
        return RNA_type
    def _reset_parameters(self, length):
        # Reset parameters to their default values
        self.flat_dim = int(self.model_config.hidden_dim * length + self.model_config.RNA_dim)
        self.last_layer = nn.Linear(self.flat_dim, self.model_config.nb_classes).to(device)

    def __repr__(self):
        return f"    (maxpool): {self.maxpool}\n" \
            f"    (embedding_layer): {self.embedding_layer}\n" \
            f"    (activation): {self.activation}\n" \
            f"    (dropout): {self.dropout}\n" \
            f"    (last_layer): {self.last_layer}\n" \
            f"    (sigmoid): {self.sigmoid}\n" \
            f")"
    def forward(self, block_out, RNA_type):
        self._reset_parameters(self.length)
        embedding_output = self.embedding_layer(self.mergelncRNA(RNA_type))#n*4
        out = torch.cat((block_out, embedding_output), dim=1)
        out = self.activation(out)
        out = self.dropout(out)
        digit = self.last_layer(out)
        pred = self.sigmoid(digit)
        return pred



class FullPLM(nn.Module):
    def __init__(self, block : object, model_config : Dict = None):
        self.model_config = model_config
        super(FullPLM, self).__init__()
        #This is the adaptor that can load different PLMs
        self.block = block
        self.localizationhead = LocalizationHead(self.model_config)
        

    def _reset_head(self, length):
        # Reset parameters to their default values
        self.localizationhead = LocalizationHead(self.model_config, length).to(device)

        
    def forward(self, tokens_ids : Tensor, x_mask : Tensor, RNA_type : Tensor):
        model_out = self.block(tokens_ids, x_mask)
        batch_size = model_out.size(0)
        length = model_out.size(2)
        self._reset_head(length)
        #flatten the embeddings
        out = model_out.reshape(batch_size, -1)
        pred = self.localizationhead(out, RNA_type)
        return pred





