#This script was used to train different model in a lightweighted mode
#The aim is to test whether the embedding is represented to the target

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
import os
import torch
import sys
sys.path.append("../")
from utils.trainer import Trainner
from BERTLocRNA.models.base_model import CustomizedModel

sys.path.append("../BERTLocRNA") #ensure the BERTLocRAN is on my sys path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(config_path=f"../conf/train_config", config_name="train_config" ,version_base=None)#identify the config from conf path
def train(cfg : DictConfig):
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
        )
    # mkdir output_dir 
    os.makedirs(f'{cfg.output_dir}/checkpoints/', exist_ok=True)
    print("output dir of this job:", cfg.output_dir)
    # initialize wandb and document the configuration
    run = wandb.init(**cfg.wandb, dir = cfg.output_dir, config = cfg)
    # save the config use this task to specific path
    OmegaConf.save(cfg, f"{cfg.output_dir}/train_config.yaml")

    #loading the model
    model = hydra.utils.instantiate(cfg.model, config = cfg.model.config).to(device)

    #loading the dataset
    train_loader, val_loader, test_loader = hydra.utils.instantiate(cfg.data) # instantiate dataloaders, should return tensor

    #instantiate the trainner
    trainer = Trainner(model = model)

    #trainning the data
    trainer.train(train_loader, val_loader, test_loader, cfg.params.epochs, cfg.params.load_checkpoint)
    
    # test 
    trainer.test(test_loader, overwrite=False)

if __name__ == "__main__":
    train()
    





    
    

