from typing import List, Dict, Tuple, Union
import json
import torch.nn as nn
import torch
from torch import Tensor
import os
import numpy as np
import torch.nn.functional as F
from torchvision.models.feature_extraction import create_feature_extractor

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

class baseclass:
    '''
    The main dispatcher for different embedder, including instantiaze and getting embedding
    '''
    def __init__(self, *args, **kwargs):
        self.from_pretrained(*args, **kwargs)
    def from_pretrained(self, *args, **kwargs):
        raise NotImplementedError
    def output(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, sequences : Union[List[str], List[int]], return_tensors = "pt", *args, **kwargs):
        return self.output(sequences, return_tensors = return_tensors, *args, **kwargs)



class ParnetTokenizer(baseclass):
    def from_pretrained(self, model_path, *args, **kwargs) -> Tuple[List]:
        model_path = os.path.join(model_path, "tokenizer",  "parnet.json")
        try:
            with open(model_path, 'r') as file:
                self.vocabulary = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.pad_token_id = 0
        
    def output(self, sequences : List[str], return_tensors = "pt", max_length = 8000) -> List[int]:
        assert len(sequences) >= 2, "Please ensure that the number of input sequences should larger than 2"
        encoding_keys = list(self.vocabulary.keys())
        tokens_list = []
        for seq in sequences:
            tokens = [encoding_keys.index(i) for i in seq]
            tokens_list.append(tokens)
        if return_tensors == "pt":
            padded_sequences = [F.pad(torch.tensor(seq), pad=(0, max_length - len(seq))) for seq in tokens_list]
            # Convert to a tensor
            tokens_list = torch.stack(padded_sequences)
            masks = [[1] * len(seq[seq!=0]) for seq in tokens_list]


        else:
            raise ValueError("you must set return_tensors to get the consistent tensor shape")


        return tokens_list, masks
    
class Parnet_model(nn.Module):
    def __init__(self, model_path, fine_tune_layers = None, ckp_name = None, output_layer = None, conditions = None):
        super(Parnet_model, self).__init__()
        self.tokenizer = ParnetTokenizer(model_path)
        self.vocabulary = self.tokenizer.vocabulary
        self.output_layer = output_layer
        self.conditions = conditions
        self.ckp_path = os.path.join(model_path, "checkpoints",  str(ckp_name))
        self.parnet_model = torch.load(self.ckp_path)
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.vocabulary.keys()),embedding_dim=len(list(self.vocabulary.values())[0]),_weight=torch.tensor(list(self.vocabulary.values())))
        self.pointwise = nn.Conv1d(500, 256, kernel_size=(1,), stride=(1,), bias=False)
        total_layers = len([i for i in self.parnet_model.named_parameters()])
        if fine_tune_layers is None:
            self.parnet_model.eval()
        else:
            freeze_index = total_layers - fine_tune_layers
            for i, (name, param) in enumerate(self.parnet_model.named_parameters()):
                if i < freeze_index:
                    param = param.to(torch.float32)
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        if output_layer is not None:
            print("initializing the model")
            for name, param in self.parnet_model.named_parameters():
                param = param.to(torch.float32)
                param.requires_grad = False
            self.print_number_of_trainable_model_parameters(self.parnet_model)
               
            #output the model in the intermediate layer
            self.parnet_model = self.wrapper_as_prob(self.parnet_model, output_layer)
            #trim the model
            print("after trim the model")
            self.print_number_of_trainable_model_parameters(self.parnet_model)
            #freeze other layers
            for param in self.embedding_layer.parameters():
                param.requires_grad = False
            for param in self.pointwise.parameters():
                param.requires_grad = False
            

        if conditions is not None:
            before = self.reset_module(self.parnet_model, "body.tower.0.0.conv1d")
            self.parnet_model.apply(self.reset_parameters)
            after = self.reset_module(self.parnet_model, "body.tower.0.0.conv1d")
            if before != after:
                print(f"########The weights of the layers have already been reset!!! methods: {self.conditions}########")
            else:
                print(f"########The weights of the layers have not been reset!!! methods: {self.conditions}########")
            print("after reset the parameters")
            self.print_number_of_trainable_model_parameters(self.parnet_model)
    def print_number_of_trainable_model_parameters(self, model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")
    
    def wrapper_as_prob(self, model, output_layer):
        l1 = 0
        l2 = 0
        if output_layer <= 6:
            l2 = output_layer - 1  
        else:
            if output_layer == 7:
                l1 += 1
            l2 = output_layer - 7
        print("output the following output_layer in the tower:", f"body.tower.{l1}.{l2}")
        prob = create_feature_extractor(model, return_nodes={f"body.tower.{l1}.{l2}": "output"})
        return prob
    def reset_module(self, model, name):
        for module_name, module in model.named_modules():
            if module_name == name:
                a = [w.norm(1) for name, w in module.named_parameters()]
                return a[0]

    def reset_parameters(self, m):
        '''
        reset the parameters of all the layers
        '''
        if isinstance(m, nn.Conv1d):
            # torch.nn.init.normal_(m.weight.data)
            if self.conditions == "samedist":
                mean = m.weight.data.mean()
                std = m.weight.data.std()
                return torch.nn.init.normal_(m.weight.data, mean=mean, std=std)
            elif self.conditions == "reinit":
                return torch.nn.init.normal_(m.weight.data)
            else:
                print(self.conditions)
                return m.weight.data
    
    def _reset_point(self, in_shape):
        # Reset parameters to their default values
        self.pointwise = nn.Conv1d(in_shape, 256, kernel_size=(1,), stride=(1,), bias=False).to(device)

    def forward(self, x : Tensor) -> Tensor:
        x = x.long()
        embedding = self.embedding_layer(x) #[batch, T, 4]
        embedding = embedding.transpose(1,2)#[batch, 4, T] 
        embedding = embedding.to(torch.float32)
        embedding = self.parnet_model(embedding)["output"]#[batch, X, T]
    
        return embedding



class ParnetModel(baseclass):
    def from_pretrained(self, model_path : str, fine_tune_layers = None, ckp_name = None, *args, **kwargs):
        print("ckp_name in module", ckp_name)
        self.model = Parnet_model(model_path, fine_tune_layers, ckp_name).to(device)
        print("after loading the model")
        
    
    def output(self, sequences : Tensor, *args, **kwargs) -> Tensor:
        assert isinstance(sequences, Tensor), "sequences must be of type Tensor"
        embedding = self.model(sequences) #[batch, 256, T]
        return embedding