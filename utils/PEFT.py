#We should rewrite some codes to fit the lora

from dataclasses import field, dataclass
from typing import List, Tuple, Dict, Union, Sequence
import torch
from torch import nn
import numpy as np
import sys
sys.path.append("../")
from BERTLocRNA.utils.embedding_generator import *



@dataclass
class DataCollatorLora(object):
    def __init__(self, max_seq_len):
        super().__init__()
        self.max_seq_len = max_seq_len
    def multi_label(self, label: List[List[str]]) -> torch.Tensor:
        # print("labels", label)
        return torch.tensor([[int(i) for i in x] for x in label])

    def trim(self, seq):
        if len(seq) >= self.max_seq_len:
            return seq[:self.max_seq_len/2, self.max_seq_len/2:]
        else:
            return seq

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        
        input_ids, masks, RNA_type, labels, ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "attention_mask", "RNA_type", "labels", "ids")
        )
        input_ids_t = self.trim(input_ids)
        input_ids_p = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=0
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
        # import pdb; pdb.set_trace()
        return dict(
            embedding=input_ids_p,
            RNA_type=RNA_type,
            labels=labels,
            attention_mask=masks,
        )#keep the embeddings the be consistent with the embedder tasks


# https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
class NucleotideTransformerLora(NucleotideTransformerEmbedder):
    """
    Embed using the Nuclieotide Transformer (NT) model https://www.biorxiv.org/content/10.1101/2023.01.11.523679v2.full
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        
        input_ids = tokens["input_ids"].int()
        masks = tokens["attention_mask"].int()

        output = {"input_ids" : input_ids, "attention_mask" : masks}#array with variant lengths
        
        return output

    def process(self, dataset :Union[DatasetDict, None]):

       
        tokenized_datasets = dataset.map(self.tokenization, batched = True, batch_size = self.batch_size)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets = tokenized_datasets.rename_column("Xtag", "RNA_type")
        tokenized_datasets.set_format("torch")
        if self.dataloader:
            data_collator = DataCollatorLora()
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
   


