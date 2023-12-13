from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, DataCollatorWithPadding, AutoModel
import torch
from dataclasses import field, dataclass
from typing import List, Tuple, Dict, Union, Sequence
import click
import os
import json
from datasets import load_dataset, DatasetDict
import numpy as np
import logging
import sys
import torch.nn as nn
sys.path.append("../../")
from torch.utils.data import DataLoader
from torch import Tensor
from BERTLocRNA.RBPLLM.Parnet import ParnetModel, ParnetTokenizer
import time
import shutil
import fm
import torch
root_dir =  os.getcwd()

device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_join = lambda *path: os.path.abspath(os.path.join(*path))


#define a overall class for loading a certain embedding
class baseclass:
    '''
    The main dispatcher for different embedder, including instantiaze and getting embedding
    '''
    def __init__(self, *args, **kwargs):
        self.embedder_name = None
        self.load_model(*args, **kwargs)
    def load_model(self, *args, **kwargs):
        raise NotImplementedError
    def process(self, *args, **kwargs):
        raise NotImplementedError
    def filter_sequences(self, seqs, effective_length):
        truncated = {}
        seq_modified = []
        for idx,seq in enumerate(seqs):
            if len(seq) > effective_length:
                seq_modified.append(seq[:effective_length])
                truncated[str(idx)] = seq[effective_length:]
            else:
                seq_modified.append(seq)

        return seq_modified, truncated
    def alignment(self, left_embeds : np.ndarray, 
                        truncated_embeds : np.ndarray,
                        left_masks : np.ndarray,
                        truncated_masks : np.ndarray,
                        truncated_d : Dict) -> List:
        '''
        make the embedding align the sequence length according to the mask, this is important for design the data collator
        '''
        count = 0
        embedding_out = []
        mask_out = []
        for idx, left_embed in enumerate(left_embeds):
            left_mask = left_masks[idx]
            if str(idx) in truncated_d.keys():
                #remove the padding
                seq_length1 = np.sum(left_mask)
                #filter out the cls token
                left_embed = left_embed[1:seq_length1]
                left_mask = left_mask[1:seq_length1]
                
                right_embed = truncated_embeds[count]
                right_mask = truncated_masks[count]
                #removing the padding
                seq_length2 = np.sum(right_mask)
                #removing the cls
                right_embed = right_embed[1:seq_length2]
                right_mask = right_mask[1:seq_length2]
                if len(left_embed.shape) >= 2:
                    new_embed = np.vstack([left_embed, right_embed])
                    embedding_out.append(new_embed)
                else:
                    new_embed = np.hstack([left_embed, right_embed])
                    embedding_out.append(list(new_embed))
                new_mask = np.hstack([left_mask, right_mask]) 
                mask_out.append(list(new_mask))
                count+=1
            else:
                seq_length1 = np.sum(left_mask)
                left_embed = left_embed[1:seq_length1]
                left_mask = left_mask[1:seq_length1]
                if len(left_embed.shape) >= 2:
                    embedding_out.append(left_embed)
                else:
                    embedding_out.append(list(left_embed))
                mask_out.append(list(left_mask))

        return embedding_out, mask_out
    
    def get_embed(self, *args, **kwargs):
        raise NotImplementedError

    def merge_input(self, left, truncated) -> Tensor:

        #gain embedding for left
        left_ids = left["input_ids"].int()
        left_masks = left["attention_mask"].int()
        left_embeds = self.get_embed(left_ids, left_masks)
        
        #gain embedding for truncated
        truncated_ids = truncated["input_ids"].int()
        truncated_masks = truncated["attention_mask"].int()
        truncated_embeds = self.get_embed(truncated_ids, truncated_masks)

        # print(left_masks.shape, left_embeds.shape, truncated_masks.shape, truncated_embeds.shape)
        #concatenate truncate with the truncated_dict
        return left_embeds, truncated_embeds, left_masks.detach().cpu().numpy(), truncated_masks.detach().cpu().numpy()

    

    def __call__(self, dataset :Union[DatasetDict, None], *args, **kwargs):
        return self.process(dataset)



#define the datacollator
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: object
    hidden_dim: 4
    pool_size: 8
    RNA_order = ["UNK", "A", "C", "G", "T", "N", "Y RNA", "lincRNA", "lncRNA", "mRNA", "miRNA", "ncRNA", "pseudo", "rRNA", "scRNA", "scaRNA", "snRNA", "snoRNA", "vRNA"]

    def multi_label(self, label: List[List[str]]) -> torch.Tensor:
        # print("labels", label)
        return torch.tensor([[int(i) for i in x] for x in label])


    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # start_time = time.time()
        embeddings, masks, RNA_type, labels, ids = tuple(
            [instance[key] for instance in instances] for key in ("embedding", "attention_mask", "RNA_type", "labels", "ids")
        )
        # print([i.shape for i in embeddings])
        # print([len(i) for i in masks])

        embeddings = torch.nn.utils.rnn.pad_sequence(
            embeddings, batch_first=True, padding_value=0
        )
        if embeddings.shape[2] == self.hidden_dim:
            embeddings = embeddings.transpose(1, 2)
        masks = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        ).float()
        if masks.shape[1] > embeddings.shape[2]:
            masks = masks[:, 1:]
        # print("embeddings shape", embeddings.shape, embeddings)
        # print("masks shape", masks.shape, masks)
        
        # Downsample the mask
        pooling = nn.MaxPool1d(self.pool_size, stride=self.pool_size)
        masks = torch.unsqueeze(pooling(masks), dim=1)

        # Get length by mask
        labels = self.multi_label(labels)
        # Merge lincRNA and lncRNA as lncRNA
        
        idx = np.where(RNA_type == self.RNA_order.index("lincRNA"))[0]
        if len(idx) > 0:
            RNA_type[idx] = self.RNA_order.index("lncRNA")
        # import pdb; pdb.set_trace()
        RNA_type = torch.tensor(RNA_type).long()
        end_time = time.time()
        # elapsed_time = end_time-start_time

        # print(f"Total time taken for loading one batch: {elapsed_time} seconds")
        return dict(
            embedding=embeddings,
            RNA_type=RNA_type,
            labels=labels,
            attention_mask=masks,
        )





# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NucleotideTransformerEmbedder(baseclass):
    """
    Embed using the Nuclieotide Transformer (NT) model https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
    """
    def load_model(self, model_path : str, batch_size : int = 8, dataloader : bool = False, hidden_dim : int = 4, pool_size : int = 8, **kwargs):

        self.embedder_name = "NT"
        local_path = path_join(root_dir, "..", "saved_model", self.embedder_name)
        if not os.path.exists(local_path):
            if 'v2' in model_path:
                self.model = AutoModelForMaskedLM.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.max_seq_len = 12282 # "model_max_length": 2048 * 6mer --> 12,288
                self.max_tokens = 2048
                self.v2 = True
            else:
                self.model = AutoModel.from_pretrained(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.max_seq_len = 5994 # "model_max_length": 1000, 6-mer --> 6000
                self.max_tokens = 1000
                self.v2 = False
            print("Creating the path ", local_path)
            os.makedirs(local_path, exist_ok = True)
            self.model.save_pretrained(local_path, push_to_hub = False)
            
        else:
            print(local_path, " already exists, loading the model locally")
            if 'v2' in model_path:
                self.model = AutoModelForMaskedLM.from_pretrained(local_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.max_seq_len = 12282 # "model_max_length": 2048, --> 12,288
                self.max_tokens = 2048
                self.v2 = True
            else:
                self.model = AutoModel.from_pretrained(local_path)
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.max_seq_len = 5994 # "model_max_length": 1000, 6-mer --> 6000
                self.max_tokens = 1000
                self.v2 = False
   
        self.model.eval()
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size


        
    def get_embed(self, tokens_ids, attention_mask) -> np.ndarray:
        tokens_ids = tokens_ids.to(device)
        attention_mask = attention_mask.to(device)
        self.model = self.model.to(device)

        if self.v2:
            outs = self.model(tokens_ids, output_hidden_states=True)['hidden_states'][-1].detach().cpu().numpy()
        else:
            outs = self.model(tokens_ids,
                              attention_mask=attention_mask,
                              encoder_attention_mask=attention_mask,
                              output_hidden_states=True
                             )['last_hidden_state'].detach().cpu().numpy() # get last hidden state

        return outs

    

    
    

    def segment_embedder(self, sample : DatasetDict) -> Dict:
        '''
        The input is a batch of sequences, which allows for faster preprocessing.
        Sequence longer than longest positional embedding should be truncated, the maximun supported sequence length should be 6*1002, which means two segements should be enough because the input sequence is 8000 nt.
        '''
        sequences = sample["Xall"]
        # Break down the sequence into segments, and ducument the truncated sequences
        seq_modified, truncated_d = self.filter_sequences(sequences, self.max_seq_len)
        
        left = self.tokenizer(
                    seq_modified,
                    truncation = True,
                    padding = "max_length",
                    return_tensors="pt"
                )
        if len(truncated_d) > 0:
            truncated = self.tokenizer(
                        list(truncated_d.values()),
                        truncation = True,
                        padding = "max_length",
                        return_tensors="pt"
                    )

        
            left_embeds, truncated_embeds, left_masks, truncated_masks = self.merge_input(left, truncated)
        
        
            embedding, masks = self.alignment(left_embeds, truncated_embeds, left_masks, truncated_masks, truncated_d)
            #Align the token with the attention mask
            input_ids, masks = self.alignment(left["input_ids"].detach().cpu().numpy(), 
                                    truncated["input_ids"].detach().cpu().numpy(), 
                                    left["attention_mask"].detach().cpu().numpy(), 
                                    truncated["attention_mask"].detach().cpu().numpy(), truncated_d)
        else:
            input_ids = left["input_ids"].int()
            masks = left["attention_mask"].int()
            embedding = self.get_embed(input_ids, masks)

        output = {"input_ids" : input_ids, "embedding" : embedding, "attention_mask" : masks}#array with variant lengths
        
        return output

    def process(self, dataset :Union[DatasetDict, None]):
        save_path = os.path.join("/", "tmp", "erda", "BERTLocRNA", "embeddings", self.embedder_name + "embedding")
        if not os.path.exists(save_path):
            tokenized_datasets = dataset.map(self.segment_embedder, batched = True, batch_size = self.batch_size)
            tokenized_datasets.save_to_disk(save_path)
        else:
            print("loading the dataset...")
            tokenized_datasets = load_dataset(save_path)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.rename_column("Xtag", "RNA_type")
        # cache_directory = os.path.expanduser("~/.cache")
        # try:
        #     # Attempt to remove the cache directory
        #     shutil.rmtree(cache_directory)
        #     print(f"Successfully removed {cache_directory}")
        # except Exception as e:
        #     print(f"Error: {e}")
        tokenized_datasets.set_format("torch")
        if self.dataloader:
            data_collator = DataCollatorForSupervisedDataset(self.tokenizer, self.hidden_dim, self.pool_size)
            train_dataloader = DataLoader(
                                            tokenized_datasets["train"], shuffle=True, batch_size=self.batch_size, collate_fn=data_collator
                                        )
            eval_dataloader = DataLoader(
                                            tokenized_datasets["validation"], batch_size=self.batch_size, collate_fn=data_collator
                                        )
            test_dataloader = DataLoader(
                                            tokenized_datasets["test"], batch_size=self.batch_size, collate_fn=data_collator
                                        )
            return train_dataloader, test_dataloader, eval_dataloader
        else:
            return tokenized_datasets



class ParnetEmbedder(baseclass):

    def load_model(self, model_path : str, 
                        batch_size : int = 8, 
                        dataloader : bool = False, 
                        max_length : int = 8000, 
                        hidden_dim : int = 4,
                        pool_size : int = 8,
                        **kwargs):
        self.tokenizer = ParnetTokenizer(model_path)
        self.model = ParnetModel(model_path)
        self.embedder_name = "Parnet"
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
    def get_embed(self, tokens_ids : Tensor) -> np.ndarray:
        tokens_ids = tokens_ids.to(device)
        outs = self.model(tokens_ids).detach().cpu().numpy() # get last hidden state

        return outs
    def segment_embedder(self, sample : DatasetDict) -> Dict:
        sequences = sample["Xall"]
        # Break down the sequence into segments, and ducument the truncated sequences

        input_ids, masks = self.tokenizer(
                    sequences,
                    return_tensors="pt"
                )

        embedding = self.get_embed(input_ids)

        output = {"input_ids" : input_ids, "embedding" : embedding, "attention_mask" : masks}#array with variant lengths
        
        return output

    def process(self, dataset : Union[DatasetDict, None]):
        
        save_path = os.path.join("/", "tmp", "erda", "BERTLocRNA", "embeddings", self.embedder_name + "embedding")
        print("embedding will be saved at:", save_path)
        if not os.path.exists(save_path):
            tokenized_datasets = dataset.map(self.segment_embedder, batched = True, batch_size = self.batch_size)
            tokenized_datasets.save_to_disk(save_path)
        else:
            print("loading the dataset...")
            tokenized_datasets = load_dataset(save_path)
        # print("removing the cache")

        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.rename_column("Xtag", "RNA_type")
        tokenized_datasets.set_format("torch")
        
        if self.dataloader:
            data_collator = DataCollatorForSupervisedDataset(self.tokenizer, self.hidden_dim, self.pool_size)
            train_dataloader = DataLoader(
                                            tokenized_datasets["train"], shuffle=True, batch_size=self.batch_size, collate_fn=data_collator
                                        )
            eval_dataloader = DataLoader(
                                            tokenized_datasets["validation"], batch_size=self.batch_size, collate_fn=data_collator
                                        )
            test_dataloader = DataLoader(
                                            tokenized_datasets["test"], batch_size=self.batch_size, collate_fn=data_collator
                                        )
            return train_dataloader, test_dataloader, eval_dataloader
        else:
            return tokenized_datasets


class RNAFMEmbedding(baseclass):
    def load_model(self, model_path : str, 
                        batch_size : int = 8, 
                        dataloader : bool = False, 
                        max_length : int = 8000, 
                        hidden_dim : int = 4,
                        pool_size : int = 8,
                        **kwargs):
        self.model,self.alphabet = fm.pretrained.rna_fm_t12(path_join(model_path, "RNA-FM_pretrained.pth"))
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embedder_name = "RNAFM"
        self.batch_size = batch_size
        self.dataloader = dataloader
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.model.eval() 

    def segment_embedder(self, sample : DatasetDict) -> Dict:
        data = [(id, seq) for id, seq in zip(sample["ids"], sample["Xall"])]
        sequences = sample["Xall"]
        # Break down the sequence into segments, and ducument the truncated sequences
        seq_modified, truncated_d = self.filter_sequences(sequences, self.max_seq_len)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        embedding = results["representations"][12]

        output = {"input_ids" : batch_tokens, "embedding" : embedding, "attention_mask" : masks}#array with variant lengths
        
        return output

    def process(self, dataset : Union[DatasetDict, None]):
        
        save_path = os.path.join("/", "tmp", "erda", "BERTLocRNA", "embeddings", self.embedder_name + "embedding")
        print("embedding will be saved at:", save_path)
        if not os.path.exists(save_path):
            tokenized_datasets = dataset.map(self.segment_embedder, batched = True, batch_size = self.batch_size)
            tokenized_datasets.save_to_disk(save_path)
        else:
            print("loading the dataset...")
            tokenized_datasets = load_dataset(save_path)
        # print("removing the cache")

        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.rename_column("Xtag", "RNA_type")
        tokenized_datasets.set_format("torch")
        
        if self.dataloader:
            data_collator = DataCollatorForSupervisedDataset(self.tokenizer, self.hidden_dim, self.pool_size)
            train_dataloader = DataLoader(
                                            tokenized_datasets["train"], shuffle=True, batch_size=self.batch_size, collate_fn=data_collator
                                        )
            eval_dataloader = DataLoader(
                                            tokenized_datasets["validation"], batch_size=self.batch_size, collate_fn=data_collator
                                        )
            test_dataloader = DataLoader(
                                            tokenized_datasets["test"], batch_size=self.batch_size, collate_fn=data_collator
                                        )
            return train_dataloader, test_dataloader, eval_dataloader
        else:
            return tokenized_datasets

#Class DM3Loc


@click.command()
@click.option("-t", "--tool", type = str, default = "NT", help = "The name of the tool you want to use to get the embeddings")
def main(tool):
    # sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
    # emb = embedgenerator(tool = tool, sequences = sequences)
    # embedding = emb.NTgenerator()
    # print(embedding.shape)
    dataset = load_dataset("TerminatorJ/localization_multiRNA")
    #getting the embedding from NT
    tokenized_datasets = NucleotideTransformerEmbedder.get_embed(dataset, batch_size = 2)



if __name__ == "__main__":
    main()