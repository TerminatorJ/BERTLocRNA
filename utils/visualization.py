#########################
#For UMAP visualization##
#########################

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



#########################
#####For RNA Sections####
#########################


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
    new = 0
    length = 0
    for token in tokens:
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
    def __init__(self, embedding : str = "Parnet", ref_path = "/home/sxr280/BERTLocRNA/data/Test_fold0_utr_cds.csv", 
                 hidden_dim = 256, batch_size: int = 10, split = "train", cache_dir = "/tmp/erda/BERTLocRNA/cache", 
                 test = True, plot = True, count = 5, tokenization = "one-hot", task = "RNAlocalization", machine = "cpu"):
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
        if machine == "cpu":
            self.device = "cpu"
        elif machine == "gpu":
            self.device = "gpu"
    def getutrcsd(self, embeddings, tokens, ids):
        embeds_5utr = []
        embeds_3utr = []
        embeds_cds = []
        random_lst = []
        if self.tokenization == "6mer":
            scale = 6
        else:
            scale = 1
        # import pdb; pdb.set_trace()
        for idx, id in enumerate(ids):
            
            utr5_s = self.ref[self.ref['ids'] == id]["a5utr_start"].values
            utr5_e = self.ref[self.ref['ids'] == id]["a5utr_end"].values
            cds_s = self.ref[self.ref['ids'] == id]["cds_start"].values
            cds_e = self.ref[self.ref['ids'] == id]["cds_end"].values
            utr3_s = self.ref[self.ref['ids'] == id]["a3utr_start"].values
            utr3_e = self.ref[self.ref['ids'] == id]["a3utr_end"].values

            embedding = embeddings[idx]
            if len(utr5_s) > 0 and not np.isnan(utr5_s) :
                if self.tokenization == "BPE":
                    # print(tokens[idx])
                    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
                    utr5_s = new_order(tokenizer, tokens[idx], utr5_s)
                    utr5_e = new_order(tokenizer, tokens[idx], utr5_e)
                    cds_s = new_order(tokenizer, tokens[idx], cds_s)
                    cds_e = new_order(tokenizer, tokens[idx], cds_e)
                    utr3_s = new_order(tokenizer, tokens[idx], utr3_s)
                    utr3_e = new_order(tokenizer, tokens[idx], utr3_e)
                if embedding.shape[0] == self.hidden_dim:
                    length = embedding.shape[1]
                    embed_5utr = embedding[:,int(utr5_s/scale):int(utr5_e/scale)].mean(axis = 1).astype('float32')
                    embed_cds = embedding[:,int(cds_s/scale):int(cds_e/scale)].mean(axis = 1).astype('float32')
                    embed_3utr = embedding[:,int(utr3_s/scale):int(utr3_e/scale)].mean(axis = 1).astype('float32')
                    # print(embed_5utr.shape)
                    # import pdb; pdb.set_trace()
                    random_number = random.sample([i for i in range(1,length)], int(utr5_e/scale)-int(utr5_s/scale))
                    random_sites = embedding[:, random_number].mean(axis = 1).astype('float32')
                    

                else:
                    length = embedding.shape[0]
                    # import pdb; pdb.set_trace()
                    embed_5utr = embedding[int(utr5_s/scale):int(utr5_e/scale), :].mean(axis = 0).astype('float32')
                    embed_cds = embedding[int(cds_s/scale):int(cds_e/scale), :].mean(axis = 0).astype('float32')
                    embed_3utr = embedding[int(utr3_s/scale):int(utr3_e/scale), :].mean(axis = 0).astype('float32')
                    # import pdb; pdb.set_trace()

                    random_number = random.sample([i for i in range(1,length)], int(utr5_e/scale)-int(utr5_s/scale))
                    random_sites = embedding[random_number, :].mean(axis = 0).astype('float32')

                embeds_5utr.append(embed_5utr)
                embeds_3utr.append(embed_3utr)
                embeds_cds.append(embed_cds)
                random_lst.append(random_sites)
            else:
                
                NAN = np.full((1,), np.nan, dtype='float32')
                embeds_5utr.append(NAN)
                embeds_cds.append(NAN)
                embeds_3utr.append(NAN)
                random_lst.append(NAN)

        return embeds_5utr, embeds_cds, embeds_3utr, random_lst


    def process_loc(self, sample):
        embedding = sample["embedding"]
        tokens = sample["input_ids"]
        ids = sample["ids"]
        subembeds = self.getutrcsd(embedding, tokens, ids)
        # import pdb; pdb.set_trace()
        embed1d_5utr = subembeds[0]
        embed1d_cds = subembeds[1]
        embed1d_3utr = subembeds[2]
        random = subembeds[3]
        output = {"embed1d_5utr" : embed1d_5utr, "embed1d_cds" : embed1d_cds, "embed1d_3utr" : embed1d_3utr, "random" : random}
        return output
    def process_RNA(self, sample):
        
        # import pdb; pdb.set_trace()
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

        #getting the dataframe
        assert len(flt_all) > 0, "The UTRs and CDS sequences extracted are empty"
        df = pd.DataFrame(flt_all)
        df_sp = sp.csr_matrix(df)
        # Save the sparse matrix as a .mtx file
        io.mmwrite(f'../data/{self.task}_{self.embedding}_{self.split}_feature_table.mtx', df_sp)
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
        fig.figure.savefig(f"/home/sxr280/BERTLocRNA/output/RNAlocalization/Figure/{self.task}_{self.embedding}_{self.split}_umap.png", dpi = 300)


    def run(self):
        dataset_name = f"/tmp/erda/BERTLocRNA/embeddings/{self.task}_{self.embedding}embedding"

        if self.test:
            start = 500
            end = 500 + self.count
            tokenized_datasets = load_dataset(dataset_name, split= f"{self.split}[{start}:{end}]", cache_dir =  self.cache_dir)

        else:

            tokenized_datasets = load_dataset(dataset_name, split= f"{self.split}", cache_dir =  self.cache_dir)
        if self.machine == "cpu":
            tokenized_datasets.set_format("numpy")
        elif self.machine == "gpu":
            tokenized_datasets.set_format("torch")
        if self.task == "RNAlocalization":
            tokenized_datasets = tokenized_datasets.map(self.process_loc, batched = True, batch_size = self.batch_size)
            feature_table = self.aforead_process_loc(tokenized_datasets)
        elif self.task == "RNAembedding":
            tokenized_datasets = tokenized_datasets.map(self.process_RNA, batched = True, batch_size = self.batch_size)
            feature_table = self.aforead_process_RNA(tokenized_datasets)

        if self.plot:
            self.plot_umap(feature_table)


    def __call__(self) -> Any:
        return  self.run()

if __name__  == "__main__":
    umap = UMAP_plot(embedding = "DNABERT2", ref_path = "/home/sxr280/BERTLocRNA/data/Train_fold0_utr_cds.csv", hidden_dim = 768,
                     batch_size = 5, split = "train", test = False, plot=True, count = 10, tokenization = "BPE", machine = "cpu", task = "RNAembedding")
    umap()



