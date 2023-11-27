import hydra
from omegaconf import DictConfig, OmegaConf
import sys
sys.path.append("../")
from BERTLocRNA.utils.embedding_generator import NucleotideTransformerEmbedder
from datasets import load_dataset, DatasetDict



@hydra.main(config_path="../conf/embed_config/", config_name="embed_config", version_base=None)
def get_embedding(cfg : DictConfig):

    print("running the embedding for task:", cfg.task)
    print("running the model:", cfg.model)
    embedder = hydra.utils.instantiate(cfg[cfg.model])
    #generating the embedding using different embedders
    #loading the dataset for a certain task
    dataset = load_dataset(**cfg[cfg.task])
    #generating the embedding and save them
    tokenized_datasets = embedder(dataset, batch_size = cfg.batch_size)

if __name__ == "__main__":
    get_embedding()

    

