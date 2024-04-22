#########################
#For UMAP visualization##
#########################
'''
In this scripts, we build a saved_embeddings -> umap_image pipe that can plot the umap according to the seperated UTR and CDS sequence
This is important, because we  calculate umap based on the input sequence of the task, which would be more flexible without repeatly calculate the embeddings

First, you need to use the "extract_utr_cds.R" to calculate the utr and cds sequence
Then run this pipeline to get the umap.
'''
import re
from typing import Any
import pandas as pd
import numpy as np
import umap
import umap.plot
from datasets import load_dataset
import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
import scipy.io as io
import pickle
import random
from transformers import AutoTokenizer
import torch
import argparse
from torch.nn.utils.rnn import pad_sequence


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def convert_tokens_to_length(tokenizer, token):
    # print("token:", token)
    string = np.array(tokenizer.convert_ids_to_tokens([token]))
    # print(string)
    #remove [CLS], [PAD], [SEP]
    string = string[~np.isin(string,["[CLS]", "[PAD]", "[SEP]"])]
    if len(string) == 0:
        out = 0
    else:
        out = len(string[0])
    return out

def new_order(tokenizer, tokens, old):
    '''
    Getting the new site in the tokenized sequences
    '''
    new = 0
    import pdb; pdb.set_trace()
    length = 0
    for token in tokens:
        #filter the tokens to skip special tokens and get the length
        token_len = convert_tokens_to_length(tokenizer, token)
        length += token_len
        if length >= old:
            new += 1
            break
        else:
            if token_len > 0:
                new += 1
    return new

    



class UMAP_plot:
    '''
    We use the dataset to match the reference path, which was generated to have utr and cds info according to the ensemble bioMart
    
    
    '''
    def __init__(self, embedding : str = "Parnet", ref_path = "/home/sxr280/BERTLocRNA/data/Test_fold0_utr_cds.csv", 
                 hidden_dim = 256, batch_size: int = 10, split = "train", cache_dir = "/tmp/erda/BERTLocRNA/cache", 
                 test = True, plot = True, count = 5, tokenization = "one-hot", task = "RNAlocalization", machine = "cpu",
                 dataset_name = "/tmp/erda/BERTLocRNA/embeddings/RNAembedding_NTembedding", layer = None, num_proc = 4):
        #if you have tokenization = BPE, you should have a model path to point out the tokenizer
        self.ref = pd.read_table(ref_path, sep=",")#TODO genome wide to keep more samples
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.ref_path = ref_path
        self.split = None
        self.batch_size = batch_size
        self.split = split
        self.cache_dir = cache_dir
        self.test = test
        self.plot = plot
        self.count = count
        self.tokenization = tokenization
        self.task = task
        self.machine = machine
        self.layer = layer
        self.num_proc = num_proc
        self.dataset_name = dataset_name
        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        if machine == "cpu":
            self.device = "cpu"
        elif machine == "gpu":
            self.device = "gpu"
        #pre-compute the location
        self.id_positions = {
            id: {
                "utr5_s": self.ref[self.ref['ids'] == id]["a5utr_start"].values,
                "utr5_e": self.ref[self.ref['ids'] == id]["a5utr_end"].values,
                "cds_start": self.ref[self.ref['ids'] == id]["cds_start"].values,
                "cds_end": self.ref[self.ref['ids'] == id]["cds_end"].values,
                "utr3_s": self.ref[self.ref['ids'] == id]["a3utr_start"].values,
                "utr3_e": self.ref[self.ref['ids'] == id]["a3utr_end"].values
                # Add other positions as needed
                }
                    for id in self.ref['ids']
                }
        
    def getutrcds(self, embeddings, tokens, ids):
        
        embeds_5utr = []
        embeds_3utr = []
        embeds_cds = []
        random_lst = []
        NAN = np.full((1,), np.nan, dtype='float32')
        # NAN = torch.tensor(float('nan')).to(device)
        if self.tokenization == "6mer":
            scale = 6
        else:
            scale = 1
        # import pdb; pdb.set_trace()
        for idx, id in enumerate(ids):
            # import pdb; pdb.set_trace()
            if id in self.id_positions:
                utr5_s = self.id_positions[id]["utr5_s"]
                utr5_e = self.id_positions[id]["utr5_e"]
                cds_s = self.id_positions[id]["cds_start"]
                cds_e = self.id_positions[id]["cds_end"]
                utr3_s = self.id_positions[id]["utr3_s"]
                utr3_e = self.id_positions[id]["utr3_e"]
                embedding = embeddings[idx]
                
                if len(utr5_s) > 0 and not np.isnan(utr5_s) :
                    #switch device
                    scale = torch.tensor(scale).int().to(embedding.device)
                    utr5_s = torch.tensor(utr5_s).int().to(embedding.device)
                    utr5_e = torch.tensor(utr5_e).int().to(embedding.device)
                    cds_s = torch.tensor(cds_s).int().to(embedding.device)
                    cds_e = torch.tensor(cds_e).int().to(embedding.device)
                    utr3_s = torch.tensor(utr3_s).int().to(embedding.device)
                    utr3_e = torch.tensor(utr3_e).int().to(embedding.device)

                    if self.tokenization == "BPE":
                        utr5_s = new_order(self.tokenizer, tokens[idx], utr5_s)
                        utr5_e = new_order(self.tokenizer, tokens[idx], utr5_e)
                        cds_s = new_order(self.tokenizer, tokens[idx], cds_s)
                        cds_e = new_order(self.tokenizer, tokens[idx], cds_e)
                        utr3_s = new_order(self.tokenizer, tokens[idx], utr3_s)
                        utr3_e = new_order(self.tokenizer, tokens[idx], utr3_e)
                    if embedding.shape[0] == self.hidden_dim:
                        length = embedding.shape[1]
                        # print(embedding.device)

                        embed_5utr = embedding[:,int(utr5_s/scale):int(utr5_e/scale)].mean(dim=1)
                        embed_cds = embedding[:,int(cds_s/scale):int(cds_e/scale)].mean(dim=1)
                        embed_3utr = embedding[:,int(utr3_s/scale):int(utr3_e/scale)].mean(dim=1)
                        # random_number = random.sample([i for i in range(1,length)], int(utr5_e/scale)-int(utr5_s/scale))
                        random_indices = torch.randint(1, length, (int(utr5_e / scale) - int(utr5_s / scale),), device=embedding.device)

                        random_sites = embedding[:, random_indices].mean(dim=1)
                        

                    else:
                        # import pdb; pdb.set_trace()
                        length = embedding.shape[0]
                        # print(embedding.device)

                        embed_5utr = embedding[int(utr5_s/scale):int(utr5_e/scale), :].mean(dim=0).cpu()
                        embed_cds = embedding[int(cds_s/scale):int(cds_e/scale), :].mean(dim=0).cpu()
                        embed_3utr = embedding[int(utr3_s/scale):int(utr3_e/scale), :].mean(dim=0).cpu()
                        # random_number = random.sample([i for i in range(1,length)], int(utr5_e/scale)-int(utr5_s/scale))
                        random_indices = torch.randint(1, length, (int(utr5_e / scale) - int(utr5_s / scale),), device=embedding.device)
                        random_sites = embedding[random_indices, :].mean(dim=0).cpu()

                    embeds_5utr.append(embed_5utr)
                    embeds_3utr.append(embed_3utr)
                    embeds_cds.append(embed_cds)
                    random_lst.append(random_sites)
                else:
                    embeds_5utr.append(NAN)
                    embeds_cds.append(NAN)
                    embeds_3utr.append(NAN)
                    random_lst.append(NAN)
            else:
                embeds_5utr.append(NAN)
                embeds_cds.append(NAN)
                embeds_3utr.append(NAN)
                random_lst.append(NAN)

        return embeds_5utr, embeds_cds, embeds_3utr, random_lst


    def process_loc(self, sample):
        
        padded_embeddings = pad_sequence(sample["embedding"], batch_first=True, padding_value=0)
        embeddings = padded_embeddings.to(device)
        tokens = sample["input_ids"]
        ids = sample["ids"]
        subembeds = self.getutrcds(embeddings, tokens, ids)
        embed1d_5utr = subembeds[0]
        embed1d_cds = subembeds[1]
        embed1d_3utr = subembeds[2]
        random = subembeds[3]
        output = {"embed1d_5utr" : embed1d_5utr, "embed1d_cds" : embed1d_cds, "embed1d_3utr" : embed1d_3utr, "random" : random}
        return output
    def process_RNA(self, sample):
        '''
        Getting the embedding list by averaging the sequence length
        '''
        
        embeddings = sample["embedding"]#3D
        embed_lst = []
        for embedding in embeddings:
            
            if embedding.shape[0] == self.hidden_dim:
                if self.machine == "gpu":
                    embedding = embedding.to(self.device)
                    embed = torch.mean(embedding, dim = 1, dtype = torch.float32)
                else:
                    embed = embedding.mean(axis = 1).astype('float32')
            elif embedding.shape[1] == self.hidden_dim:
                if self.machine == "gpu":
                    embedding = embedding.to(self.device)
                    embed = torch.mean(embedding, dim = 0, dtype = torch.float32)
                else:
                    embed = embedding.mean(axis = 0).astype('float32') 
            embed_lst.append(embed) 
        output = {"embeddings": embed_lst}
        return output
        


    def getRNA(self, id):
        pattern = r"RNA_category:([^,\n]+)"
        RNA_type = re.findall(pattern, id)[0]
        return RNA_type
    
    def aforead_process_loc(self,tokenized_datasets):
        embed1d_5utr = tokenized_datasets["embed1d_5utr"]
        embed1d_3utr = tokenized_datasets["embed1d_3utr"]
        embed1d_cds = tokenized_datasets["embed1d_cds"]
        random = tokenized_datasets["random"]
        #getting the RNA types
        RNA_5utr = [self.getRNA(i) for i in tokenized_datasets["ids"]]
        RNA_3utr = [self.getRNA(i) for i in tokenized_datasets["ids"]]
        RNA_cds =  [self.getRNA(i) for i in tokenized_datasets["ids"]]
        RNA_random = [self.getRNA(i) for i in tokenized_datasets["ids"]]

        #hstack three compartments together
        embed2d = np.hstack([embed1d_5utr, embed1d_3utr, embed1d_cds, random])
        flt_lst = []
        drop_idx = []
        labels = len(embed1d_5utr) * ["5UTR"] + len(embed1d_3utr) * ["3UTR"] + len(embed1d_cds) * ["CDS"] + len(random) * ["Random"]
        Rlabels = RNA_5utr + RNA_3utr + RNA_cds + RNA_random
        labels_f = []
        Rlabels_f = []
        for idx,i in enumerate(embed2d):
            if not np.isnan(i).any():
                flt_lst.append(i)
                labels_f.append(labels[idx])
                Rlabels_f.append(Rlabels[idx])
            else:
                drop_idx.append(idx)
        #concat array
        flt_all = np.vstack(flt_lst)
        print(f"{len(drop_idx)} samples have been filtered")
        print(f"{len(flt_lst)} samples have been kept")

        #getting the dataframe
        assert len(flt_all) > 0, "The UTRs and CDS sequences extracted are empty"
        df = pd.DataFrame(flt_all)
        df_sp = sp.csr_matrix(df)
        # Save the sparse matrix as a .mtx file
        io.mmwrite(f'../data/{self.task}_{self.embedding}_{self.split}_{self.layer}_feature_table.mtx', df_sp)
        pickle.dump(labels_f, open(f"../data/{self.task}_{self.embedding}_{self.split}_labels.pkl", "wb"))
        pickle.dump(Rlabels_f, open(f"../data/{self.task}_{self.embedding}_{self.split}_Rlabels.pkl", "wb"))
        df["Labels"] = labels_f
        df["RLabels"] = Rlabels_f
        
        return df
    def aforead_process_RNA(self,tokenized_datasets):
        embeddings = tokenized_datasets["embeddings"]
        labels = tokenized_datasets["label"]
        # import pdb; pdb.set_trace()
        df = pd.DataFrame(embeddings)
        df_sp = sp.csr_matrix(df)
        io.mmwrite(f'../data/{self.task}_{self.embedding}_{self.split}_feature_table.mtx', df_sp)
        pickle.dump(labels, open(f"../data/{self.task}_{self.embedding}_{self.split}_labels.pkl", "wb"))
        # pickle.dump(Rlabels_f, open(f"../data/{self.task}_{self.embedding}_{self.split}_Rlabels.pkl", "wb"))
        df["Labels"] = labels
        return df
        


    def plot_umap(self, df, n_neighbors = 10, min_dist = 0.1):
        mapper = umap.UMAP(n_neighbors = n_neighbors, min_dist = min_dist).fit(df.loc[:, ~df.columns.isin(["Labels", "RLabels"])])
        fig = umap.plot.points(mapper, labels=df["Labels"], color_key_cmap='tab20')
        fig.figure.savefig(f"/home/sxr280/BERTLocRNA/output/RNAlocalization/Figure/{self.task}_{self.embedding}_{self.split}_{self.layer}_umap.png", dpi = 300)


    def run(self):
        # dataset_name = f"/tmp/erda/BERTLocRNA/embeddings/{self.task}_{self.embedding}embedding"

        if self.test:
            start = 500
            end = 500 + self.count
            tokenized_datasets = load_dataset(self.dataset_name, split= f"{self.split}[{start}:{end}]", cache_dir =  self.cache_dir)

        else:

            tokenized_datasets = load_dataset(self.dataset_name, split= f"{self.split}", cache_dir =  self.cache_dir)
        if self.machine == "cpu":
            tokenized_datasets.set_format("numpy")
        elif self.machine == "gpu":
            tokenized_datasets.set_format("torch", columns = ["embedding"])
        if self.task == "RNAelements":
            tokenized_datasets = tokenized_datasets.map(self.process_loc, batched = True, batch_size = self.batch_size, num_proc = self.num_proc)
            feature_table = self.aforead_process_loc(tokenized_datasets)
        elif self.task == "RNAtypes":
            tokenized_datasets = tokenized_datasets.map(self.process_RNA, batched = True, batch_size = self.batch_size, num_proc = self.num_proc)
            feature_table = self.aforead_process_RNA(tokenized_datasets)

        if self.plot:
            self.plot_umap(feature_table)


    def __call__(self) -> Any:
        return  self.run()

if __name__  == "__main__":
    #When calculating the umap with RNA species labels
    # runumap = UMAP_plot(embedding = "DNABERT2", ref_path = "/home/sxr280/BERTLocRNA/data/Train_fold0_utr_cds.csv", hidden_dim = 768,
    #                  batch_size = 5, split = "train", test = True, plot=True, count = 10, tokenization = "BPE", machine = "cpu", task = "RNAtypes",
    #                  dataset_name = "/tmp/erda/BERTLocRNA/embeddings/RNAembedding_NTembedding")
    # runumap()

    parser = argparse.ArgumentParser(description='Run UMAP plot')
    parser.add_argument('--layer', type=int, required=True, help='The layer to use')
    args = parser.parse_args()
    #When calculating the umap with rna functional elements
    runumap = UMAP_plot(embedding = "NT", ref_path = "/home/sxr280/BERTLocRNA/data/Train_fold0_utr_cds.csv", hidden_dim = 2560,
                     batch_size = 500, split = "train", test = False, plot=True, count = 10, tokenization = "6mer", machine = "gpu", task = "RNAelements",
                     dataset_name = f"/tmp/erda/BERTLocRNA/embeddings/RNAlocalization_probe_NTembedding_{args.layer}transfer", layer = args.layer, num_proc = 1)
    runumap()




