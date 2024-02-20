##This is an implementation of fine-tuning PLM using LoRA adaption
##19/2/2024

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, DataCollatorWithPadding, AutoModel
import os
import torch
from torch import nn
from typing import List, Tuple, Dict, Union, Sequence
from torch import Tensor
from BERTLocRNA.models.layers import *

class Lora:
    def __init__(self, max_tokens = None, lora_config = None):
        self.max_tokens = max_tokens
        self.lora_config = lora_config

    def wrapper(self, model):
        lora_config = LoraConfig(
                r=self.lora_config["r"], # Rank
                lora_alpha=self.lora_config["lora_alpha"],
                target_modules=self.lora_config["target_modules"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS # BERT
            )
        peft_model = get_peft_model(model, lora_config)
        return peft_model
    @staticmethod
    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"



class LoraModel(Lora):
    def __init__(self, PLM : object = None, model_config : Dict = None, lora_config : Dict = None):
        self.model_config = model_config
        self.lora_config = lora_config
        if PLM is not None:
            self.PLM = PLM()
            #Calculating the effeciency of the LoRA in the PLM
            Lora.print_number_of_trainable_model_parameters(self.PLM.model)
            super(Lora).__init__(max_tokens = self.PLM.max_tokens, lora_config = lora_config)
            flat_dim = self.hidden_dim*self.PLM.max_tokens/model_config.pooling_size + model_config.RNA_dim
            self.last_layer = nn.Linear(flat_dim , model_config.nb_classes)
        else:
            super(Lora).__init__()

        
        #the input need to be truncated again to fixed the length limitation
        #we should add the last layer to fit for the multi-label prediction
        self.maxpool = nn.MaxPool1d(model_config.pooling_size, stride = model_config.pooling_size)
        self.embedding_layer = nn.Embedding(num_embeddings=len(model_config.RNA_order),embedding_dim=model_config.RNA_dim,_weight=torch.zeros((len(model_config.RNA_order), model_config.RNA_dim)))
        self.sigmoid = nn.Sigmoid()
        self.Actvation = Actvation(model_config.activation)
        self.dropout = nn.Dropout(model_config.drop_flat)



    def mergelncRNA(self, RNA_type):
        idx = torch.where(RNA_type == self.model_config.RNA_order.index("lincRNA"))[0]
        if len(idx) > 0:
            RNA_type[idx] = self.model_config.RNA_order.index("lncRNA")
        return RNA_type
        
    def forward(self, embed : Tensor, x_mask : Tensor, RNA_type : Tensor):
        PLM_out = self.PLM.get_embed(embed, x_mask)
        batch_size = PLM_out.size(0)
        #flatten the embeddings
        PLM_out = torch.view(batch_size, -1)
        embedding_output = self.embedding_layer(self.mergelncRNA(RNA_type))#n*4
        out = torch.cat((PLM_out, embedding_output), dim=1)
        out = self.Actvation(out)
        out = self.dropout(out)
        digit = self.last_layer(out)
        pred = self.sigmoid(digit)
        return pred





