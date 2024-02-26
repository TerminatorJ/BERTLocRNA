#We should rewrite some codes to fit the lora
#When embedding a new model to the architechture, we should add one class in this module, and one block module in the layers module


from dataclasses import field, dataclass
from typing import List, Tuple, Dict, Union, Sequence
import torch
from torch import nn
import numpy as np
import sys
sys.path.append("../")
from BERTLocRNA.utils.embedding_generator import *
from BERTLocRNA.utils.layers import *
import pickle
from BERTLocRNA.RBPLLM.Parnet import ParnetModel, ParnetTokenizer, Parnet_model
hf_cache = "/tmp/erda/BERTLocRNA/cache"




class DataCollatorLora(object):
    def __init__(self, max_seq_len, remove_CLS = False, remove_SEP = False):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.remove_CLS = remove_CLS
        self.remove_SEP = remove_SEP
        self.RNA_order = ["UNK", "A", "C", "G", "T", "N", "Y RNA", "lincRNA", "lncRNA", "mRNA", "miRNA", "ncRNA", "pseudo", "rRNA", "scRNA", "scaRNA", "snRNA", "snoRNA", "vRNA"]
    def multi_label(self, label: List[List[str]]) -> torch.Tensor:
        return torch.tensor([[int(i) for i in x] for x in label])

    def trim(self, seq):
        if len(seq) >= self.max_seq_len:
            half_len = self.max_seq_len/2
            trimmed_seq = torch.cat([seq[:int(half_len)], seq[-int(half_len):]])
            return trimmed_seq
        else:
            return seq

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, masks, RNA_type, labels, ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "attention_mask", "RNA_type", "labels", "ids")
        )
        if self.remove_CLS:
            input_ids = [i[1:] for i in input_ids]
            masks = [i[1:] for i in masks]
        if self.remove_SEP:
            input_ids = [i[:torch.sum(j)-1] for i,j in zip(input_ids, masks)]
            masks = [i[:torch.sum(i)-1] for i in masks]
        input_ids_t = list(map(lambda x:  self.trim(x), input_ids))
        input_ids_p = torch.nn.utils.rnn.pad_sequence(
            input_ids_t, batch_first=True, padding_value=0
        )
        masks = torch.nn.utils.rnn.pad_sequence(
            masks, batch_first=True, padding_value=0
        ).float()
        # Get length by mask
        labels = self.multi_label(labels)
        # Merge lincRNA and lncRNA as lncRNA
        idx = np.where(RNA_type == self.RNA_order.index("lincRNA"))[0]
        if len(idx) > 0:
            RNA_type[idx] = self.RNA_order.index("lncRNA")
        RNA_type = torch.tensor(RNA_type).long()
        return dict(
            embedding=input_ids_p,
            RNA_type=RNA_type,
            labels=labels,
            attention_mask=masks,
        )#keep the embeddings the be consistent with the embedder tasks




# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NTModuleTokenizer(NucleotideTransformerEmbedder):
    def __init__(self, lora_config, run_batch = 1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = NT_blcok(lora_config, self.model, self.hidden_dim)
        self.run_batch = run_batch
        
    def tokenization(self, sample : DatasetDict) -> Dict:
        '''
        The input is a batch of sequences, which allows for faster preprocessing.
        Sequence longer than longest positional embedding should be truncated, the maximun supported sequence length should be 6*1002, which means two segements should be enough because the input sequence is 8000 nt.
        '''
        sequences = sample["Xall"]
        # Break down the sequence into segments, and ducument the truncated sequences
        tokens = self.tokenizer(
                    sequences,
                    truncation = True,
                    padding = "max_length",
                    return_tensors="pt"
                )
        #filter out the cls token
        input_ids = tokens["input_ids"].int()
        masks = tokens["attention_mask"].int()
        output = {"input_ids" : input_ids, "attention_mask" : masks}#array with variant lengths
        return output


    def __call__(self, dataset :Union[DatasetDict, None]):

        save_file = os.path.join("/", "tmp", "erda", "BERTLocRNA", "embeddings", self.task + "_" + self.embedder_name + "embedding.pkl")
        print("processing################################")
        
        if not os.path.isfile(save_file):
            tokenized_datasets = dataset.map(self.tokenization, batched = True, batch_size = self.run_batch)
            pickle.dump(tokenized_datasets, open(save_file, "wb"))
        else:
            print("loading the dataset...")
            tokenized_datasets = pickle.load(open(save_file, "rb"))

        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.rename_column("Xtag", "RNA_type")
        tokenized_datasets.set_format("torch")
        if self.dataloader:
            data_collator = DataCollatorLora(self.max_seq_len, remove_CLS = True)
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
        
   


class DNABERT2ModuleTokenizer(DNABERT2Embedder):
    def __init__(self, lora_config, run_batch = 1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = DNABERT2_blcok(lora_config, self.model, self.hidden_dim)
        self.run_batch = run_batch
        


    def tokenization(self, sample : DatasetDict) -> Dict:
        '''
        The input is a batch of sequences, which allows for faster preprocessing.
        Sequence longer than longest positional embedding should be truncated, the maximun supported sequence length should be 6*1002, which means two segements should be enough because the input sequence is 8000 nt.
        '''
        sequences = sample["Xall"]
        #Tokenize the sequences
        tokens_ids = self.tokenizer(sequences, return_tensors = 'pt', padding=True)["input_ids"]
        #Getting masks
        masks = [[1]*len(id[(id != self.cls_id) & (id != self.sep_id) & (id != self.pad_id)]) for id in tokens_ids]
        output = {"input_ids" : tokens_ids, "attention_mask" : masks}#array with variant lengths
        
        return output


    def __call__(self, dataset :Union[DatasetDict, None]):

        save_file = os.path.join("/", "tmp", "erda", "BERTLocRNA", "embeddings", self.task + "_" + self.embedder_name + "embedding.pkl")
        print("processing################################")
        
        if not os.path.isfile(save_file):
            tokenized_datasets = dataset.map(self.tokenization, batched = True, batch_size = self.run_batch)
            pickle.dump(tokenized_datasets, open(save_file, "wb"))
            # tokenized_datasets.save_to_disk(save_path)
        else:
            print("loading the dataset...")
            tokenized_datasets = pickle.load(open(save_file, "rb"))

        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.rename_column("Xtag", "RNA_type")
        tokenized_datasets.set_format("torch")
        if self.dataloader:
            data_collator = DataCollatorLora(self.max_seq_len, remove_CLS = True, remove_SEP = True)
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
        

class RNAFMModuleTokenizer(RNAFMEmbedder):
    def __init__(self, lora_config, run_batch = 1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block = RNAFM_blcok(lora_config, self.model, self.hidden_dim)
        self.run_batch = run_batch

    def tokenization(self, sample : DatasetDict) -> Dict:
        '''
        The input is a batch of sequences, which allows for faster preprocessing.
        '''
        sequences = sample["Xall"]
        pair = [(id, seq) for id, seq in zip(sample["ids"], sequences)]
        tokens = self.tokenizer(pair)[2]

        masks = [[1]*len(seq) for seq in sequences]

        output = {"input_ids" : tokens, "attention_mask" : masks}
        return output


    def __call__(self, dataset :Union[DatasetDict, None]):

        save_file = os.path.join("/", "tmp", "erda", "BERTLocRNA", "embeddings", self.task + "_" + self.embedder_name + "embedding.pkl")
        print("processing################################")
        
        if not os.path.isfile(save_file):
            tokenized_datasets = dataset.map(self.tokenization, batched = True, batch_size = self.run_batch)
            pickle.dump(tokenized_datasets, open(save_file, "wb"))
            # tokenized_datasets.save_to_disk(save_path)
        else:
            print("loading the dataset...")
            tokenized_datasets = pickle.load(open(save_file, "rb"))

        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.rename_column("Xtag", "RNA_type")
        tokenized_datasets.set_format("torch")
        if self.dataloader:
            data_collator = DataCollatorLora(self.max_seq_len, remove_CLS = True, remove_SEP = True)
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



class parnetModuleTokenizer(ParnetEmbedder):
    def __init__(self, lora_config, run_batch = 1024, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Parnet_model(kwargs["model_path"], fine_tune_layers=kwargs["fine_tune_layers"])
        self.block = Parnet_blcok(lora_config, self.model, self.hidden_dim)
        self.run_batch = run_batch
        
    def tokenization(self, sample : DatasetDict) -> Dict:
        '''
        The input is a batch of sequences, which allows for faster preprocessing.
        '''
        sequences = sample["Xall"]
        input_ids, masks = self.tokenizer(
                    sequences,
                    return_tensors="pt"
                )

        output = {"input_ids" : input_ids, "attention_mask" : masks}
        return output


    def __call__(self, dataset :Union[DatasetDict, None]):

        save_file = os.path.join("/", "tmp", "erda", "BERTLocRNA", "embeddings", self.task + "_" + self.embedder_name + "embedding.pkl")
        print("processing################################")
        
        if not os.path.isfile(save_file):
            tokenized_datasets = dataset.map(self.tokenization, batched = True, batch_size = self.run_batch)
            pickle.dump(tokenized_datasets, open(save_file, "wb"))
            # tokenized_datasets.save_to_disk(save_path)
        else:
            print("loading the dataset...")
            tokenized_datasets = pickle.load(open(save_file, "rb"))

        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.rename_column("Xtag", "RNA_type")
        tokenized_datasets.set_format("torch")
        if self.dataloader:
            data_collator = DataCollatorLora(self.max_seq_len)
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



