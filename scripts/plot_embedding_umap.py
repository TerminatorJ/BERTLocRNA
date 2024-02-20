#This script was used to train different model in a lightweighted mode
#The aim is to test whether the embedding is represented to the target

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys
sys.path.append("../")
sys.path.append("../..")
from utils import visualization

os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path=f"../conf/visualization_config", config_name="vis_config" ,version_base=None)#identify the config from conf path
def plot(cfg : DictConfig):

    # save the config use this task to specific path
    OmegaConf.save(cfg, f"{cfg.output_dir}/vis_config.yaml")
    umap = hydra.utils.instantiate(cfg.umap)
    #making plot
    umap()

if __name__ == "__main__":
    plot()
    





    
    

