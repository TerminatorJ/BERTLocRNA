from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, DataCollatorWithPadding
import torch
from dataclasses import field, dataclass
from typing import List, Tuple, Dict, Union
import click
import os
import json
from datasets import load_dataset, DatasetDict
import numpy as np
import logging
path_join = lambda *path: os.path.abspath(os.path.join(*path))
root_dir =  os.getcwd()

@dataclass
class embedgenerator:
    tool : str = field(default = "NT", metadata = {"help" : {"The foundation model you want to launch"}})
    def __post_init__(self):
        #loading the model configuration
        with open(path_join(root_dir, "model.json"), 'r') as json_file:
            model_kwargs = json.load(json_file)

        model_path = path_join(root_dir, "model", self.tool)
        

        if not os.path.exists(model_path):
            self.tokenizer = AutoTokenizer.from_pretrained(**model_kwargs[self.tool])
            self.model = AutoModelForMaskedLM.from_pretrained(**model_kwargs[self.tool])
            print("Creating the path ", model_path)
            os.makedirs(model_path, exist_ok = True)
            self.model.save_pretrained(model_path, push_to_hub = False)
            
        else:
            print(model_path, " already exists, loading the model locally")
            self.tokenizer = AutoTokenizer.from_pretrained(**model_kwargs[self.tool])
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)
            
        #ensuring the max position embedding is larger than max_length
        
        config = AutoConfig.from_pretrained(model_path)
        self.max_length = config.max_position_embeddings
        logging.warning(f"The maximum token length of this model is: {self.max_length}")

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
    
    def merge(self, id_all, id_truncated, truncated_dict : Dict, key) -> np.ndarray:
        count = 0
        final_list = []
        
        for idx, input_id in enumerate(id_all[key]):
            if idx in truncated_dict.keys():
                new_id = input_id + id_truncated[key][count]
                final_list.append(new_id)
                count+=1
            else:
                final_list.append(input_id)
        
        return final_list



    def segment_tokenizer(self, sample : DatasetDict, kmer = 6):
        '''
        The input is a batch of sequences, which allows for faster preprocessing.
        Sequence longer than longest positional embedding should be truncated, the maximun supported sequence length should be 6*1002, which means two segements should be enough.
        '''
        sequences = sample["Xall"]
        # Break down the sequence into segments
        effective_length = kmer * self.max_length
        
        seq_modified, truncated = self.filter_sequences(sequences, effective_length)
        # print(type(seq_modified))
        
        ids_all = self.tokenizer(
                    seq_modified,
                    truncation = True,
                )
        id_truncated = self.tokenizer(
                    list(truncated.values()),
                    truncation = True
                )
        final_merge = list(map(lambda x: self.merge(ids_all, id_truncated, truncated, x), ["input_ids","attention_mask"]))
        output = dict(input_ids = final_merge[0], attention_mask = final_merge[1])
        
        return output

    @classmethod
    def NTgenerator(cls, kmer, dataset :Union[DatasetDict, None]):
        embed = cls(tool = "NT")
        tokenized_datasets = dataset.map(embed.segment_tokenizer, batched = True)
        print(tokenized_datasets)
        return tokenized_datasets
        #building data collator
        # data_collator = DataCollatorWithPadding(tokenizer = embed.tokenizer)
        # attention_mask = embed.tokens_ids != embed.tokenizer.pad_token_id
        # torch_outs = self.model(
        #     self.tokens_ids,
        #     attention_mask=attention_mask,
        #     encoder_attention_mask=attention_mask,
        #     output_hidden_states=True
        # )


        # # Compute sequences embeddings
        # embeddings = torch_outs['hidden_states'][-1].detach().numpy() # (n,T,2560)
        # return embeddings
    
    #TODO DNABERT2
    def DNABERT2(self):
        
        embeddings = self.model(self.tokens_ids)[0] # [1, sequence_length, 768]

        return embeddings



@click.command()
@click.option("-t", "--tool", type = str, default = "NT", help = "The name of the tool you want to use to get the embeddings")
def main(tool):

    dataset = load_dataset("TerminatorJ/localization_multiRNA")
    #getting the embedding from NT
    embeding = embedgenerator.NTgenerator(kmer = 6, dataset = dataset)


    # print(tool = tool, dataset)

if __name__ == "__main__":
    main()