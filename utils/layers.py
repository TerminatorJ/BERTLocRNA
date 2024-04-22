from typing import List, Tuple, Dict, Union, Sequence
import torch
from torch import nn
import numpy as np
import sys
sys.path.append("../")
from BERTLocRNA.utils.embedding_generator import *
from peft import LoraConfig, get_peft_model, TaskType
from torchinfo import summary
import torch.nn.functional as F
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


class NT_block(nn.Module):
    def __init__(self, lora_config = None, model = None, hidden_dim = None, output_layer = None, conditions = "transfer"):
        
        print("initializing NT block")
        super().__init__()
        self.model = model.to(device)
        #set to the train mode
        self.model.train()
        self.lora = Lora(lora_config)
        self.model = self.lora.wrapper(self.model)

        if output_layer is not None:
            for name, param in self.model.named_parameters():
                param = param.to(torch.float32)
                param.requires_grad = False

        print("#############PEFT wrapped model#############")
        self.model.print_trainable_parameters()
        # self.lora.print_number_of_trainable_model_parameters(self.model)
        self.hidden_dim = hidden_dim
        self.output_layer = self.layer_alignment(output_layer)
        
        self.conditions = conditions
        self.tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-2.5b-multi-species")

        if self.conditions != "transfer":
            before = self.reset_module(self.model, "base_model.model.encoder.layer.0.attention.self.query")
            # print("before:", before)
            self.model.apply(self.reset_parameters)
            after = self.reset_module(self.model, "base_model.model.encoder.layer.0.attention.self.query")
            # print("after:", after)
            if before != after:
                print(f"########The weights of the layers have already been reset!!! methods: {self.conditions}########")
            else:
                print(f"########The weights of the layers have not been reset!!! methods: {self.conditions}########")
 
    def reset_module(self, model, name):
        for module_name, module in model.named_modules():
            # print(module_name)
            if module_name == name:
                a = [w.norm(1) for name, w in module.named_parameters()]
                return a[0]
    def wrapper_as_probe(self):

        for layer, (name, param) in enumerate(self.model.named_parameters()):
            param.requires_grad = False
        # print(next(iter(self.model.base_model.encoder.layer[self.output_layer].parameters())))
        # Unfreeze parameters before the the certain layer

        for i in range(self.output_layer+1):
            for param in self.model.base_model.encoder.layer[i].parameters():
                param.requires_grad = True
        # print(next(iter(self.model.base_model.encoder.layer[self.output_layer].parameters())))

    def layer_alignment(self, output_layer):
        hidden_layers = [0, 4, 8, 12, 16, 19, 22, 25, 28, 32] 
        return hidden_layers[output_layer-1]

        #make the model output to be a certain layer
    def reset_parameters(self, m):
        '''
        reset the parameters of all the layers
        '''
        if isinstance(m, nn.Linear):
            # torch.nn.init.normal_(m.weight.data)
            
            if self.conditions == "samedist":
                mean = m.weight.data.mean()
                std = m.weight.data.std()
                torch.nn.init.normal_(m.weight.data, mean=mean, std=std)
            elif self.conditions == "reinit":
                # print("here is the reinit")
                torch.nn.init.normal_(m.weight.data)
            else:
                pass

    def forward(self, tokens_ids : Tensor, x_mask : Tensor):
        tokens_ids = tokens_ids.to(torch.long)

        if self.output_layer == None:
            outs = self.model(tokens_ids,
                                        attention_mask=x_mask,
                                        encoder_attention_mask=x_mask,
                                        output_hidden_states=True
                                        )["hidden_states"][-1]
            #calibrate the dimension
            outs = calibrate(outs, self.hidden_dim)
        else:
            # print("output layer:", self.output_layer)
            self.wrapper_as_probe()
            outs = self.model(tokens_ids,
                            attention_mask=x_mask,
                            encoder_attention_mask=x_mask,
                            output_hidden_states=True
                            )["hidden_states"][self.output_layer]
            # print("out shape", outs.shape)
            outs =  calibrate(outs, self.hidden_dim)
        return outs

class DNABERT2_block(nn.Module):
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
    def forward(self, tokens_ids : Tensor, x_mask : Tensor):
        tokens_ids = tokens_ids.to(torch.long)
        outs = self.model(tokens_ids)[0]# [batch, sequence_length, 768]
        #calibrate the dimension
        outs = calibrate(outs, self.hidden_dim)
        return outs


class RNAFM_block(nn.Module):
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
    def forward(self, tokens_ids : Tensor, x_mask : Tensor):
        tokens_ids = tokens_ids.to(torch.long)
        results = self.model(tokens_ids, repr_layers=[12])
        outs = results["representations"][12]
        #calibrate the dimension
        outs = calibrate(outs, self.hidden_dim)
        return outs
    
class Parnet_block(nn.Module):
    def __init__(self, lora_config = None, model = None, hidden_dim = None, output_layer = None):
        
        print("Initializing Parnet block")
        super().__init__()
        self.model = model.to(device)
        #set to the train mode
        if output_layer == None:
            self.model.train()
        self.lora = Lora(lora_config)
        self.lora.print_number_of_trainable_model_parameters(self.model)
        self.hidden_dim = hidden_dim
        self.maxpool = nn.MaxPool1d(8000)
    def forward(self, tokens_ids : Tensor, x_mask : Tensor):
        tokens_ids = tokens_ids.to(torch.long)
        outs = self.model(tokens_ids)
        #calibrate the dimension
        outs = calibrate(outs, self.hidden_dim)
        outs = self.maxpool(outs)
        return outs


