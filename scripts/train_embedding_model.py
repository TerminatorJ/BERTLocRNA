#This script was used to train different model in a lightweighted mode
#The aim is to test whether the embedding is represented to the target

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import torch
import sys
sys.path.append("../")
sys.path.append("../..")
from BERTLocRNA.utils.trainer import MyTrainer
from datasets import load_dataset, Value, Features
from BERTLocRNA.utils.optional import Weights

os.environ["HYDRA_FULL_ERROR"] = "1"
#saving the cache file to ERDA
os.environ["HF_HOME"] = "/tmp/erda/BERTLocRNA/cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path=f"../conf/train_conf", config_name="train_config" ,version_base=None)#identify the config from conf path
def train(cfg : DictConfig):

    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
        )
    # initialize wandb and document the configuration
    print("initializing")
    wandb.init(**cfg.wandb, dir = cfg.output_dir, config = cfg)
    # wandb.config._service_wait = 100

    os.makedirs(cfg.output_dir , exist_ok = True)
    print("output dir of this job:", cfg.output_dir)
    
    
    #making the directory to save everything
    
    # save the config use this task to specific path
    OmegaConf.save(cfg, f"{cfg.output_dir}/train_config.yaml")

    #loading the model
    model = hydra.utils.instantiate(cfg.base_model)
    #loading the dataset
    embedder = hydra.utils.instantiate(cfg[cfg.embedder])
    #generating the embedding using different embedders 
    custom_features = Features({"label": Value(dtype='string'), 
                                "idx": Value(dtype="int64"),
                                "Xall": Value(dtype='string'),
                                "Xtag": Value(dtype="int64"),
                                "ids": Value(dtype='string')})
    #loading the dataset for a certain task
    dataset = load_dataset(**cfg[cfg.task], features=custom_features)
    #Calculating the classweight, even though weighted result not gain performance
    if cfg.loss_weight:
        weight = Weights(cfg.nb_classes, cfg.sample_t)
        #getting the weight dict and the labels that are filtered according to abundance
        weight_dict, flt_dict, pos_weight = weight.get_weight(dataset["train"]["Xtag"], dataset["train"]["label"], cfg.base_model.config.RNA_order, save_dir = f"{cfg.output_dir}")
    else:
        weight = Weights(cfg.nb_classes, cfg.sample_t)
        #getting the weight dict and the labels that are filtered according to abundance
        weight_dict, flt_dict, pos_weight = weight.get_weight(dataset["train"]["Xtag"], dataset["train"]["label"], cfg.base_model.config.RNA_order, save_dir = f"{cfg.output_dir}")
        weight_dict = None


    # import pdb; pdb.set_trace()
    #generating the embedding and save them
    train_dataloader, test_dataloader, eval_dataloader = embedder(dataset)
    
    #instantiate the trainner
    Trainer = MyTrainer(model = model, 
                        config = cfg.Trainer.config, 
                        weight_dict = None,
                        flt_dict = flt_dict)

    #trainning the data
    Trainer.train(train_dataloader, eval_dataloader)
    
    # test 
    Trainer.test(test_dataloader, flt_dict)

if __name__ == "__main__":
    train()
    





    
    

