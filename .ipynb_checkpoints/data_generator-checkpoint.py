import numpy as np
import re
from dataclasses import dataclass, field
import os
from sklearn.model_selection import KFold
import json
from typing import List, Dict, Tuple
import logging
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import click
from datasets import Dataset
import pandas as pd
#todo: fasta -> dataloader
#5-fold: optional
#tiny testing model: mandatory
#deleted pad mode
#delete pool function
#setting every time to get the RNA with tag


#define the path join function
path_join = lambda *path: os.path.abspath(os.path.join(*path))
root_dir =  os.getcwd()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Gene_data:
    def __init__(self):
        self.id_label_seq_Dict = {}
        self.label_id_Dict = {}
        self.id = []

    def load_sequence(self, dataset=None, left=1000, right=3000, predict=False, RNA_type=None):
        with open(dataset, 'r') as f:
            index=0
            for line in f:
                if line[0] == '>':
                    if index!=0:
                        seq=seq.upper()
                        seq=seq.replace('U','T')
                        seq=list(seq)
                        #change all other characters into N
                        for index in range(len(seq)):
                            if seq[index] not in ['A','C','G','T']:
                               seq[index]='N'
                        
                        seq = ''.join(seq)
                        
                        seq_length = len(seq)
                        line_left = seq[:int(seq_length*left/(right+left))]
                        line_right = seq[int(seq_length*left/(right+left)):]
                        if len(line_right) >= right:
                            line_right = line_right[-right:]
                        
                        if len(line_left) >= left:
                            line_left = line_left[:left]
                        
                        self.id_label_seq_Dict.setdefault(id, {}).setdefault(label, (line_left.strip(), line_right.strip()))
                        self.label_id_Dict.setdefault(label, []).append(id)
                        

                    
                    id = line.strip()
                    if RNA_type != "allRNA":
                        # print("this is not all RNA", RNA_type)
                        label = line[1:].split(',')[0] #changed to label not float
                    else:
                        pattern = r"RNA_category:([^,\n]+)"
                        rna_type = re.findall(pattern, line)

                        # RNA_type = line.split("RNA_category:")[1].split(",")[0].strip()
                        label = line[1:].split(',')[0] + rna_type[0]
                    seq=""
                else:
                    seq+=line.strip()
                
                #print(index)
                index+=1
            
            #last seq 
            seq=seq.upper()
            seq=seq.replace('U','T')
            seq=list(seq)
            #change all other characters into N
            for index in range(len(seq)):
                if seq[index] not in ['A','C','G','T']:
                   seq[index]='N'
            
            seq = ''.join(seq)
            
            seq_length = len(seq)
            line_left = seq[:int(seq_length*left/(right+left))]
            line_right = seq[int(seq_length*left/(right+left)):]
            if len(line_right) >= right:
                line_right = line_right[-right:]
            
            if len(line_left) >= left:
                line_left = line_left[:left]
            
            self.id_label_seq_Dict.setdefault(id, {}).setdefault(label, (line_left.strip(), line_right.rstrip()))
            self.label_id_Dict.setdefault(label, []).append(id)

    


    

@dataclass
class Locdata(Dataset):
    left : int = field(default = 4000, metadata= {"help": "The left length of the sequence"}) 
    right : int = field(default = 4000, metadata = {"help": "The right length of the sequence"})
    data_path : str = field(default = "/home/sxr280/DeepLocRNA/DeepLocRNA/data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_pooled_deduplicated3_filtermilncsnsno.fasta", metadata = {"help": "The input fasta, defined as '.fasta'"})
    foldnum : int = field(default = 5, metadata={"help": "setting the fold to split the data, if foldnum = 1, data will be split as train, test, val"})
    RNA_type : str = field(default = "allRNA", metadata={"help": "which RNA you want to extract and process, allRNA means a mixture of all kinds of RNAs"})
    save_json: bool = field(default = False, metadata = {"help": "whether saving the data as json file"})
    vocabulary_path: str = field(default = "./vocabulary.json", metadata = {"help" : "the vocabulary of RNA species and nucleotide, which can be used to encode and identify RNA types"})
    tiny_test: bool = field(default = False, metadata={"help": "generate a tiny testing dataset"})
    target_num: int = field(default = 9, metadata={"help": "number of target of the samples"})
    save_csv : bool = field(default = True, metadata = {"help" : "whether to save the file as .csv for loading the dataset"})
    id_label_seq_Dict = None
    

    
    def __post_init__(self):
        self.encoding_keys = self.get_dict()
        OUT = self.generate_data()



    def get_dict(self):
        encoding_dict = json.load(open(self.vocabulary_path, "r"))
        encoding_keys = list(encoding_dict.keys())
        return encoding_keys
        

    def generate_data(self) -> Tuple[Dict[int, Tuple[List[int], List[int], List[int], List[int]]]]:
        gene_data = Gene_data()
        gene_data.load_sequence(dataset = self.data_path, left = self.left, right = self.right, predict = False, RNA_type = self.RNA_type)
        self.id_label_seq_Dict = gene_data.id_label_seq_Dict
        label_id_Dict = gene_data.label_id_Dict

        #define the output data
        OUT = {}


       
        if os.path.exists(path_join(root_dir, "data", "Train5" + "1.json")):
            Train = {}
            Test = {}
            Val = {}
            for i in range(self.foldnum):
                Train[i] = json.load(open(path_join(root_dir, "data", f"Train5{i}.json"), "r"))
                Test[i] = json.load(open(path_join(root_dir, "data", f"Test5{i}.json"), "r"))
                Val[i] = json.load(open(path_join(root_dir, "data", f"Val5{i}.json"), "r"))

        else:

            (Train, Test, Val) = self.group_sample(label_id_Dict)
        for fold in range(self.foldnum):
            for partname, part in {"Train":Train, "Test":Test, "Val":Val}.items():
                Xall, Xtag, ids, Y = self.fold_data(part[fold])
                OUT.setdefault(partname, {}).setdefault(fold, part)
                #saving to csv
                if self.save_csv:
                    data = {
                        'idx' : [i for i in range(len(Xall))],
                        'Xall': Xall,
                        'Xtag': Xtag,
                        'ids': ids,
                        'label': Y
                    }

                    # Create a DataFrame
                    print(Y, Xall, ids)
                    df = pd.DataFrame(data)

                    # Save the DataFrame to a CSV file
                    df.to_csv(f'./data/{partname}_fold{fold}.csv', index=False)

        return OUT



    def fold_data(self, ids) -> Tuple[List[np.ndarray], List[str], List[List[int]]]:


        X_left = [[c for c in list(self.id_label_seq_Dict[id].values())[0][0]] for id in ids]
        X_right = [[c for c in list(self.id_label_seq_Dict[id].values())[0][1]] for id in ids]
        Xall = ["".join(list(np.concatenate([x,y],axis=-1))) for x,y in zip(X_left,X_right)]
        Xtag = self.get_tag(ids)
        Y = np.array([list(self.id_label_seq_Dict[id].keys())[0][:self.target_num] for id in ids])#question?

        return Xall, Xtag, ids, Y


    def label_dist(self, dist):
        label = []
        for x in dist:
            try:
                label.append(int(x))
            except:
                continue

        return label

    def get_tag(self, ids) -> List[str]:
        tags = []
        for id in ids:
            pattern = r'RNA_category:([^,\n]+)'
            RNA_types = re.findall(pattern, id)
            RNA_tag = self.encoding_keys.index(RNA_types[0])
            tags.append(RNA_tag)
        return tags

    def group_sample(self, label_id_Dict) -> Tuple[Dict[int, List[str]]]:
        Train = {}
        Test = {}
        Val = {}
        
        for eachkey in label_id_Dict:#decide which item was used to split k fold
            label_ids = np.array(label_id_Dict[eachkey])#transfer to be an array --> ndarray
            if len(label_ids) < self.foldnum:
                for i in range(self.foldnum):
                    Train.setdefault(i, []).extend(list(label_ids))
                continue
            train_fold_ids, val_fold_ids,test_fold_ids = self.typeicalSampling(label_ids)
            # print("train_fold_ids", train_fold_ids)
            for i in range(self.foldnum):
                Train.setdefault(i, []).extend(train_fold_ids[i])
                # print(Train[i])
                Val.setdefault(i, []).extend(val_fold_ids[i])
                Test.setdefault(i, []).extend(test_fold_ids[i])
                if self.tiny_test:
                    Train.setdefault(i, []).extend(train_fold_ids[i][:5])
                    Val.setdefault(i, []).extend(val_fold_ids[i][:5])
                    Test.setdefault(i, []).entend(test_fold_ids[i][:5])
                    self.save_json = False

        logging.warning(f"Finishing group sampling")
        if self.save_json:
            os.makedirs(path_join(root_dir, "data"), exist_ok = True)
            for i in range(self.foldnum):
                # print(Train[i])
                json.dump(Train[i], open(path_join(root_dir, "data", f'Train{self.foldnum}{i}.json'),"w"))
                json.dump(Test[i], open(path_join(root_dir, "data", f'Test{self.foldnum}{i}.json'),"w"))
                json.dump(Val[i], open(path_join(root_dir, "data", f'Val{self.foldnum}{i}.json'),"w"))

        return Train, Test, Val
    def typeicalSampling(self, ids : np.ndarray) -> Tuple[Dict[str, List[int]]]:
        kf = KFold(n_splits = self.foldnum, shuffle=True, random_state=1234)
        folds = kf.split(ids)
        train_fold_ids = {}
        val_fold_ids = {}
        test_fold_ids= {}
        for i, (train_indices, test_indices) in enumerate(folds):
            size_all = len(train_indices)
            train_indices2 = train_indices[:int(size_all * 0.8)]
            val_indices = train_indices[int(size_all * 0.8):]
            train_fold_ids[i] = list(ids[train_indices2])
            val_fold_ids[i] = list(ids[val_indices])
            test_fold_ids[i] = list(ids[test_indices])
        return train_fold_ids,val_fold_ids,test_fold_ids
    

#building the customized Dataset

class Locdataset(Dataset):
    def __init__(self, train_dataset : Dataset):
        self.Xall = train_dataset.Xall
        self.Xpad = train_dataset.Xpad
        self.Xtag = train_dataset.Xtag
        self.mask = train_dataset.mask
        self.ids = train_dataset.ids
        self.label = train_dataset.y
        # self.features = ["Xall", "Xpad", "Xtag", "mask", "ids", "label"]

    def __len__(self):
        return len(self.Xall)

    def __getitem__(self, idx):
        sample = {
            "Xall" :  self.Xall[idx],
            "Xpad" : self.Xpad[idx],
            "Xtag" : self.Xtag[idx],
            "mask" : self.mask[idx],
            "ids" : self.ids[idx],
            "label" : self.label[idx]

        }
        return sample
        


  




@click.command()
@click.option("-ds", "--dataset", type = str, default = "/home/sxr280/DeepLocRNA/DeepLocRNA/data/allRNA/allRNA_all_human_data_seq_mergedm3locall2_pooled_deduplicated3_filtermilncsnsno.fasta", help = "The input fasta, defined as '.fasta'")
@click.option("-fn", "--foldnum", type = int, default = 5, help = "setting the fold to split the data, if foldnum = 1, data will be split as train, test, val")
@click.option("-rna", "--rna-type", type = str, default = "allRNA", help = "which RNA you want to extract and process, allRNA means a mixture of all kinds of RNAs")
@click.option("-sj", "--save-json", type = bool, default = True, help = "whether saving the data as json file")
@click.option("-vp", "--vocabulary-path", type = str, default = "./vocabulary.json", help = "the vocabulary of RNA species and nucleotide, which can be used to encode and identify RNA types")
@click.option("-p", "--parts", type = str, default = "train", help = "the part of the split data you want to get, from ['train', 'test', 'val']")
@click.option("-fd", "--fold", type = int, default = 0, help = "The fold you want to get the data")


def main(dataset, foldnum, rna_type, save_json, vocabulary_path, parts, fold):
    dataset_0 = Locdata(data_path = dataset, foldnum = foldnum, RNA_type = rna_type, save_json = save_json, vocabulary_path = vocabulary_path, parts = parts, fold = fold, encode = False)
    for k,v in dataset_0[0].items():
        print(k, v, type(v))
  
    # torch.save(dataset_0 ,path_join(root_dir, "data", f"{parts}data_fold{fold}.pt"))

    

if __name__ == "__main__":
    main()    