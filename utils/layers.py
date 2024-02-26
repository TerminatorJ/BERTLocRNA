from typing import List, Tuple, Dict, Union, Sequence
import torch
from torch import nn
import numpy as np
import sys
sys.path.append("../")
from BERTLocRNA.utils.embedding_generator import *
from peft import LoraConfig, get_peft_model, TaskType
from torchinfo import summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calibrate(x, dim):
    #calibrate the dimension
    if x.shape[2] == dim:
        x = x.transpose(1, 2)
    return x


class Lora:
    def __init__(self, lora_config = None):
        self.lora_config = lora_config

    def wrapper(self, model = None):
        lora_config = LoraConfig(
                r=self.lora_config["r"], # Rank
                lora_alpha=self.lora_config["lora_alpha"],
                target_modules=self.lora_config["target_modules"],
                lora_dropout=0.05,
                bias="none"
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
        print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")


class NT_blcok(nn.Module):
    def __init__(self, lora_config = None, model = None, hidden_dim = None):
        
        print("initializing NT block")
        super().__init__()
        self.model = model.to(device)
        #set to the train mode
        self.model.train()
        self.lora = Lora(lora_config)
        self.model = self.lora.wrapper(self.model)
        print("#############PEFT wrapped model#############")
        self.model.print_trainable_parameters()
        # self.lora.print_number_of_trainable_model_parameters(self.model)
        self.hidden_dim = hidden_dim
    
    def forward(self, tokens_ids : Tensor, x_mask : Tensor):
        tokens_ids = tokens_ids.to(torch.long)
        outs = self.model(tokens_ids,
                                    attention_mask=x_mask,
                                    encoder_attention_mask=x_mask,
                                    output_hidden_states=True
                                    )["hidden_states"][-1]
        #calibrate the dimension
        outs = calibrate(outs, self.hidden_dim)
        return outs

class DNABERT2_block:
    def __init__(self, lora_config = None, model = None, hidden_dim = None):
        
        print("Initializing DNABERT2 block")
        super().__init__()
        self.model = model.to(device)
        #set to the train mode
        self.model.train()
        self.lora = Lora(lora_config)
        self.model = self.lora.wrapper(self.model)
        self.lora.print_number_of_trainable_model_parameters(self.model)
        self.hidden_dim = hidden_dim
    def __call__(self, tokens_ids : Tensor, x_mask : Tensor):
        tokens_ids = tokens_ids.to(torch.long)
        outs = self.model(tokens_ids)[0]# [batch, sequence_length, 768]
        #calibrate the dimension
        outs = calibrate(outs, self.hidden_dim)
        return outs


class RNAFM_blcok:
    def __init__(self, lora_config = None, model = None, hidden_dim = None):
        
        print("Initializing RNAFM block")
        super().__init__()
        self.model = model.to(device)
        #set to the train mode
        self.model.train()
        print("attribution model before lora:", dir(self.model))
        self.lora = Lora(lora_config)
        self.model = self.lora.wrapper(self.model)
        print("attribution model after lora:", dir(self.model))
        self.lora.print_number_of_trainable_model_parameters(self.model)
        self.hidden_dim = hidden_dim
    def __call__(self, tokens_ids : Tensor, x_mask : Tensor):
        tokens_ids = tokens_ids.to(torch.long)
        results = self.model(tokens_ids, repr_layers=[12])
        outs = results["representations"][12]
        #calibrate the dimension
        outs = calibrate(outs, self.hidden_dim)
        return outs
    
class Parnet_blcok:
    def __init__(self, lora_config = None, model = None, hidden_dim = None):
        
        print("Initializing Parnet block")
        super().__init__()
        self.model = model.to(device)
        #set to the train mode
        # self.model.train()
        # self.lora = Lora(lora_config)
        # self.model = self.lora.wrapper(self.model)
        # self.lora.print_number_of_trainable_model_parameters(self.model)
        self.hidden_dim = hidden_dim
    def __call__(self, tokens_ids : Tensor, x_mask : Tensor):
        tokens_ids = tokens_ids.to(torch.long)
        outs = self.model(tokens_ids)
        #calibrate the dimension
        outs = calibrate(outs, self.hidden_dim)
        return outs


