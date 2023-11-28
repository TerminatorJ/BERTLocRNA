import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import sys
sys.path.append("../../")
from BERTLocRNA.utils.embedding_generator import NucleotideTransformerEmbedder
from datasets import load_dataset, DatasetDict
import os

os.environ["HYDRA_FULL_ERROR"] = "1"

#Save all the embedding locally

@hydra.main(config_path="../conf/embed_config/", config_name="embed_config", version_base=None)
def get_embedding(cfg : DictConfig) -> None:
    print("running the embedding for task:", cfg.task)
    print("running the model:", cfg.embedder)
    embedder = hydra.utils.instantiate(cfg[cfg.embedder])
    #generating the embedding using different embedders
    #loading the dataset for a certain task
    dataset = load_dataset(**cfg[cfg.task])
    #generating the embedding and save them
    tokenized_datasets = embedder(dataset)
if __name__ == "__main__":
    get_embedding()

    

