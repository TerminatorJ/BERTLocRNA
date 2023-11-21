from data_generator import Locdata, Locdataset
import torch
from torch.utils.data import DataLoader
import os
from embedding_generator import embedgenerator
from datasets import load_dataset, DatasetDict
#TODO: 1) split the data 2) Dataset format


# path_join = lambda *path: os.path.abspath(os.path.join(*path))
# root_dir =  os.getcwd()


#Step 1: getting the data
#saving the csv file for each fold
dataobj = Locdata(data_path = "/home/sxr280/DeepLocRNA/DeepLocRNA/data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_pooled_deduplicated3_filtermilncsnsno.fasta", save_csv = True, foldnum = 5)


#saving the split data as Dataset




#step2: from dataloader to embedding
# traindata = torch.load("/home/sxr280/BERTLocRNA/data/traindata_fold0.pt")
# testdata = torch.load("/home/sxr280/BERTLocRNA/data/testdata_fold0.pt")
# valdata = torch.load("/home/sxr280/BERTLocRNA/data/valdata_fold0.pt")

# traindataset_wrapper = Locdataset(traindata)
# locdataset = load_dataset("python", data_files = traindataset_wrapper)
# print(locdataset)

# train_dataloader = DataLoader(traindata, batch_size = 8, shuffle = True)
# results = next(iter(train_dataloader))
# xtrain = results["X"]


# emb = embedgenerator(tool = "NT", sequences = xtrain, max_length = 100)
# embedding = emb.NTgenerator() 
# print(embedding.shape)