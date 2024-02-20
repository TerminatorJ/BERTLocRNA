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

@hydra.main(config_path="../conf/embed_config/", config_name="embed_config", version_base=None)
def get_embedding(cfg : DictConfig) -> None:
    print("running the embedding for task:", cfg.task)
    print("running the model:", cfg.embedder)
    embedder = hydra.utils.instantiate(cfg[cfg.embedder])
    #generating the embedding using different embedders
    #loading the dataset for a certain task
    if cfg.task == "RNAlocalization":
        custom_features = Features({"label": Value(dtype='string'), 
                                    "idx": Value(dtype="int64"),
                                    "Xall": Value(dtype='string'),
                                    "Xtag": Value(dtype="int64"),
                                    "ids": Value(dtype='string')})
        dataset = load_dataset(**cfg[cfg.task], features=custom_features)
    elif cfg.task == "RNAembedding":
        custom_features = Features({"label": Value(dtype='string'), 
                                    "Xall": Value(dtype='string'),
                                    "ids": Value(dtype='string')})
        dataset = load_dataset(**cfg[cfg.task], features=custom_features, cache_dir = "/tmp/erda/BERTLocRNA/cache")
    #generating the embedding and save them
    tokenized_datasets = embedder(dataset)
if __name__ == "__main__":
    get_embedding()

    

