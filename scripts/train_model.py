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
from utils.trainer import MyTrainer
from datasets import load_dataset

os.environ["HYDRA_FULL_ERROR"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path=f"../conf/train_conf", config_name="train_config" ,version_base=None)#identify the config from conf path
def train(cfg : DictConfig):
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
        )
    os.makedirs(cfg.output_dir , exist_ok = True)
    print("output dir of this job:", cfg.output_dir)
    
    # initialize wandb and document the configuration
    wandb.init(**cfg.wandb, dir = cfg.output_dir, config = cfg)
    #making the directory to save everything
    
    # save the config use this task to specific path
    OmegaConf.save(cfg, f"{cfg.output_dir}/train_config.yaml")

    #loading the model
    model = hydra.utils.instantiate(cfg.base_model)
    print("main function initialization:")
    for name, param in model.named_parameters():
        print("name of the model parameters:", name)
    #loading the dataset
    embedder = hydra.utils.instantiate(cfg[cfg.embedder])
    #generating the embedding using different embedders
    #loading the dataset for a certain task
    dataset = load_dataset(**cfg[cfg.task])
    #generating the embedding and save them
    train_dataloader, test_dataloader, eval_dataloader = embedder(dataset)
    
    #instantiate the trainner
    Trainer = MyTrainer(model = model, config = cfg.Trainer.config)

    #trainning the data
    Trainer.train(train_dataloader, eval_dataloader)
    
    # test 
    Trainer.test(test_dataloader)

if __name__ == "__main__":
    train()
    





    
    

