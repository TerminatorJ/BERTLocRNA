from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, DataCollatorWithPadding, AutoModel
import torch
from dataclasses import field, dataclass
from typing import List, Tuple, Dict, Union
import click
import os
import json
from datasets import load_dataset, DatasetDict
import numpy as np
import logging
from torch.utils.data import DataLoader
from utils import *
from torch import Tensor

path_join = lambda *path: os.path.abspath(os.path.join(*path))
root_dir =  os.getcwd()
# device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class TemplateEmbed:
#     '''
#     This is the template for all embedder, all of them should process the sequence in the following methods, getting the 
#     embedding in according paradims.
#     '''
    
#     def __init__(self, max_seq_len : int = None, max_tokens : int = None, tool :str = "aa"):
#         self.max_seq_len = max_seq_len
#         self.max_tokens = max_tokens
#         self.tool = tool

    
# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NucleotideTransformerEmbedder:
    """
    Embed using the Nuclieotide Transformer (NT) model https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
    """
    def __init__(self):
        model_name = "NT"
        #loading the model configuration
        with open(path_join(root_dir, "model.json"), 'r') as json_file:
            model_kwargs = json.load(json_file)
        model_path = model_kwargs["NT"]["pretrained_model_name_or_path"]
        local_path = path_join(root_dir, "model", model_name)
        if not os.path.exists(local_path):
            if 'v2' in model_path:
                self.model = AutoModelForMaskedLM.from_pretrained(**model_kwargs[model_name])
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                self.max_seq_len = 12282 # "model_max_length": 2048, --> 12,288
                self.max_tokens = 2048
                self.v2 = True
            else:
                self.model = AutoModel.from_pretrained(**model_kwargs[model_name])
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
   
        # self.model.to(device)
        self.model.eval()
        self.tool = "NT"
        self.filter_cls = True
        
    
        # super().__init__(max_seq_len = self.max_seq_len, max_tokens = self.max_tokens, tool = self.tool)
        
    def get_embed(self, tokens_ids, attention_mask) -> np.ndarray:
        if self.v2:
            outs = self.model(tokens_ids, output_hidden_states=True)['hidden_states'][-1].detach().cpu().numpy()
        else:
            # print("tokens_ids:",tokens_ids, tokens_ids.shape)
            # print("attention_mask:", attention_mask, attention_mask.shape)
            outs = self.model(tokens_ids,
                              attention_mask=attention_mask,
                              encoder_attention_mask=attention_mask,
                              output_hidden_states=True
                             )['last_hidden_state'].detach().cpu().numpy() # get last hidden state

        return outs

    
    def merge_input(self, left, truncated) -> Tensor:
        count = 0
        output_embeds = []
        output_mask = []
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
                left_mask = left_mask[:seq_length1]
                
                right_embed = truncated_embeds[count]
                right_mask = truncated_masks[count]
                #removing the padding
                seq_length2 = np.sum(right_mask)
                right_embed = right_embed[:seq_length2]
                right_mask = right_mask[:seq_length2]
                
                
                # print("left_embed", type(left_embed), left_embed.shape, "right_embed", type(right_embed), right_embed.shape)
                if len(left_embed.shape) >= 2:
                    new_embed = np.vstack([left_embed, right_embed])
                    embedding_out.append(new_embed)
                else:
                    new_embed = np.hstack([left_embed, right_embed])
                    embedding_out.append(list(new_embed))
                new_mask = np.hstack([left_mask, right_mask])
                # print("new_embed:", new_embed.shape)
                
                mask_out.append(list(new_mask))
                count+=1
            else:
                seq_length1 = np.sum(left_mask)
                left_embed = left_embed[1:seq_length1]
                left_mask = left_mask[:seq_length1]
                if len(left_embed.shape) >= 2:
                    embedding_out.append(left_embed)
                else:
                    embedding_out.append(list(left_embed))
                mask_out.append(list(left_mask))

        # print([len(i) for i in embedding_out])
        # print("2", len(embedding_out))

        return embedding_out, mask_out

    def segment_embedder(self, sample : DatasetDict) -> Dict:
        '''
        The input is a batch of sequences, which allows for faster preprocessing.
        Sequence longer than longest positional embedding should be truncated, the maximun supported sequence length should be 6*1002, which means two segements should be enough because the input sequence is 8000 nt.
        '''
        sequences = sample["Xall"]
        # Break down the sequence into segments, and ducument the truncated sequences
        seq_modified, truncated_d = filter_sequences(sequences, self.max_seq_len)
        
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
        # print("input sample size:", sample, type(sample), len(sample["Xall"]))
        # print("input_ids:", input_ids, type(input_ids), len(input_ids))
        # print("attention_mask:", masks, type(masks), len(masks))
        # print("embedding:", embedding, len(embedding), len(embedding))
        output = {"input_ids" : input_ids, "embedding" : embedding, "attention_mask" : masks}#array with variant lengths
        
        return output

    
    @classmethod
    def save_embed(cls, dataset :Union[DatasetDict, None], batch_size :int = 8):
        save_path = path_join(root_dir, "data", "NTembedding")
        if not os.path.exists(save_path):
            embed = cls()
            tokenized_datasets = dataset.map(embed.segment_embedder, batched = True, batch_size = batch_size)
            tokenized_datasets = tokenized_datasets.remove_columns(["idx", "Xall", "Xtag", "ids"])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            print("Saving to disk...")
            tokenized_datasets.save_to_disk(save_path, format="json", compression="gzip")
        else:
            print("loading the dataset...")
            tokenized_datasets = load_dataset("json", data_files={"train": f"{save_path}/train.json.gz", "val": f"{save_path}/val.json.gz", "test": f"{save_path}/test.json.gz"})

        return tokenized_datasets
        # tokenized_datasets.set_format("torch")
        # data_collator = DataCollatorWithPadding(tokenizer = embed.tokenizer)
        # train_dataloader = DataLoader(tokenized_datasets["train"], shuffle = True, batch_size = batch_size, data_collator = data_collator)
        # val_dataloader = DataLoader(tokenized_datasets["validation"], shuffle = True, batch_size = batch_size, data_collator = data_collator)
        # test_dataloader = DataLoader(tokenized_datasets["test"], shuffle = True, batch_size = batch_size, data_collator = data_collator)
        # return train_dataloader, val_dataloader, test_dataloader

        
    




@click.command()
@click.option("-t", "--tool", type = str, default = "NT", help = "The name of the tool you want to use to get the embeddings")
def main(tool):
    # sequences = ["ATTCCGATTCCGATTCCG", "ATTTCTCTCTCTCTCTGAGATCGATCGATCGAT"]
    # emb = embedgenerator(tool = tool, sequences = sequences)
    # embedding = emb.NTgenerator()
    # print(embedding.shape)
    dataset = load_dataset("TerminatorJ/localization_multiRNA")
    #getting the embedding from NT
    tokenized_datasets = NucleotideTransformerEmbedder.save_embed(dataset, batch_size = 100)



if __name__ == "__main__":
    main()