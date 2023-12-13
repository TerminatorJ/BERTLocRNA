from typing import List, Dict, Tuple, Union
import json
import torch.nn as nn
import torch
from torch import Tensor
import os
import torch.nn.functional as F

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
    def from_pretrained(self, model_path, *args, **kwargs):
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
            masks = [torch.tensor([1] * len(seq) + [0] * (max_length - len(seq))) for seq in tokens_list]
            masks = torch.stack(masks)
        else:
            raise ValueError("you must set return_tensors as pt to get the consistent tensor shape")


        return tokens_list, masks
    
class Parnet_model(nn.Module):
    def __init__(self, model_path, fine_tune_layers = None):
        super(Parnet_model, self).__init__()
        self.tokenizer = ParnetTokenizer(model_path)
        self.vocabulary = self.tokenizer.vocabulary
        self.ckp_path = os.path.join(model_path, "checkpoints",  "network.PanRBPNet.2023-03-13.ckpt")
        try:
            self.parnet_model = torch.load(self.ckp_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Checkpoint file not found: {self.ckp_path}")
        self.embedding_layer = nn.Embedding(num_embeddings=len(self.vocabulary.keys()),embedding_dim=len(list(self.vocabulary.values())[0]),_weight=torch.tensor(list(self.vocabulary.values())))
        if fine_tune_layers is None:
            self.parnet_model.eval()
        else:
            total_layers = len([i for i in self.parnet_model.named_parameters()])
            freeze_index = total_layers - fine_tune_layers
            for i, (name, param) in enumerate(self.parnet_model.named_parameters()):
                if i < freeze_index:
                    param = param.to(torch.float32)
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    
    def forward(self, x : Tensor) -> Tensor:
        x = x.long()
        embedding = self.embedding_layer(x) #[batch, T, 4]
        embedding = embedding.transpose(1,2)#[batch, 4, T] 
        embedding = embedding.to(torch.float32)
        embedding = self.parnet_model.forward(embedding)#[batch, 256, T]
        return embedding



class ParnetModel(baseclass):
    def from_pretrained(self, model_path : str, *args, **kwargs):

        self.model = Parnet_model(model_path).to(device)
        
    
    def output(self, sequences : Tensor, *args, **kwargs) -> Tensor:
        assert isinstance(sequences, Tensor), "sequences must be of type Tensor"
        embedding = self.model(sequences) #[batch, 256, T]
        return embedding