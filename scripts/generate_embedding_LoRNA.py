import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import sys
sys.path.append("../")
sys.path.append("../../")
import os
from datasets import load_dataset, Value, Features
import BERTLocRNA

os.environ["HYDRA_FULL_ERROR"] = "1"

#Save all the embedding locally

@hydra.main(config_path="../conf/embed_config/", config_name="LoRNA_embed_config", version_base=None)
def get_embedding(cfg : DictConfig) -> None:
    print("running the embedding for task:", cfg.task)
    print("running the model:", cfg.embedder)
    embedder = hydra.utils.instantiate(cfg[cfg.embedder])

    #loading the dataset for a certain task
    #we change to the cache2 because we have modified the codes
    dataset = load_dataset(**cfg[cfg.task], cache_dir = "/tmp/erda/BERTLocRNA/cache2")
    #rename the colnames to fit the dataset
    dataset = dataset.rename_column("transcripts", "Xall")
    tokenized_datasets = embedder(dataset)
if __name__ == "__main__":
    get_embedding()

    

