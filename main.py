from data_generator import dataloader_generator
import torch
from torch.utils.data import DataLoader

traindata_fold0 = torch.load("./data/traindata_fold0.pt", map_location = "cpu")
batch_size = 32
data_loader = DataLoader(traindata_fold0, batch_size=batch_size, shuffle=True)

# Iterate through batches
X, Xtag, mask, label =  next(iter(data_loader))
print(X, Xtag, mask, label)