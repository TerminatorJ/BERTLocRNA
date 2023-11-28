
import torch
import torch.nn as nn
import pytorch_lightning as pl
import time
import inspect
import re
import json
import os
import numpy as np
import sys
sys.path.append("../")
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchinfo import summary
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score,roc_auc_score,accuracy_score,matthews_corrcoef,f1_score,precision_score,recall_score
from typing import List, Tuple, Dict, Union
import pandas as pd

#make the code reproducible 
# Set random seed for NumPy
np.random.seed(42)

# Set random seed for PyTorch
torch.manual_seed(42)

# If using GPU, set random seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
gpus = torch.cuda.device_count()
path_join = lambda *path: os.path.abspath(os.path.join(*path))

class LightningModel(pl.LightningModule):
    def __init__(self, model, class_weights = None):
        super(LightningModel, self).__init__()
        #only used to extract the RNA types
        self.network = model.to(device)
        self.class_weights = class_weights
        self.epoch_start_time = None
        self.loss_fn = nn.BCELoss()

    def naive_loss(self, y_pred, y_true, ohem=False, focal=False):

        loss_weight_ = self.class_weights
        loss_weight = []
        for i in range(self.nb_classes):
        # initialize weights
            loss_weight.append(torch.tensor(loss_weight_[i],requires_grad=False, device=device))
        num_task = self.nb_classes
        num_examples = y_true.shape[0]

        def binary_cross_entropy(x, y,focal=True):
            alpha = 0.75
            gamma = 2

            pt = x * y + (1 - x) * (1 - y)
            at = alpha * y + (1 - alpha)* (1 - y)

            # focal loss
            if focal:
                loss = -at*(1-pt)**(gamma)*(torch.log(x) * y + torch.log(1 - x) * (1 - y))
            else:
                epsilon = 1e-4  # Small epsilon value
                # Add epsilon to x to prevent taking the logarithm of 0
                x = torch.clamp(x, epsilon, 1 - epsilon)
                loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
            return loss
        loss_output = torch.zeros(num_examples).to(device = device)
        for i in range(num_task):
            if loss_weight != None:
                out = loss_weight[i]*binary_cross_entropy(y_pred[:,i],y_true[:,i],focal)                
                loss_output += out
            else:
                loss_output += binary_cross_entropy(y_pred[:, i],y_true[:,i],focal)

        # Online Hard Example Mining
        if ohem:
            val, idx = torch.topk(loss_output,int(0.7*num_examples))
            loss_output[loss_output<val[-1]] = 0
        loss = torch.sum(loss_output)/num_examples
        return loss
    def binary_accuracy(self, y_pred, y_true):
        # Round the predicted values to 0 or 1
        y_pred_rounded = torch.round(y_pred)
        # Calculate the number of correct predictions
        correct = (y_pred_rounded == y_true).float().sum()
        # Calculate the accuracy
        accuracy = correct / y_true.numel()
        return accuracy
    
    def categorical_accuracy(self, y_pred, y_true):
        # Get the index of the maximum value (predicted class) along the second dimension
        y_pred = torch.argmax(y_pred, dim=1)
        y_true = torch.argmax(y_true, dim=1)
        # Compare the predicted class with the target class and calculate the mean accuracy
        return (y_pred == y_true).float().mean()

    def forward(self, embed, mask, RNA_type = None):
        x = x.to(device)
        mask = mask.to(device)
        if RNA_type != None:
            RNA_type = RNA_type.to(device)
        pred = self.network(embed, mask, RNA_type)
        return pred
    
    def configure_optimizers(self):
        optimizer = eval(self.config["optimizer_cls"])(self.parameters(), lr = self.config["lr"], weight_decay = self.config["weight_decay"])
        return optimizer

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        print(f"Epoch duration: {epoch_time:.2f} seconds")
    def _attention_regularizer(self, attention):
        batch_size = attention.shape[0]
        headnum = self.network.headnum
        identity = torch.eye(headnum).to(device)  # [r,r]
        temp = torch.bmm(attention, attention.transpose(1, 2)) - identity  # [none, r, r]
        penal = 0.001 * torch.sum(temp**2) / batch_size
        return penal

    def training_step(self, batch, **kwargs):

        x, mask, RNA_type, y = batch["embedding"], batch["attention_mask"], batch["RNA_type"], batch["labels"]
        y = y.to(torch.float32)
        y_pred = self.forward(x, mask, RNA_type)

        if self.class_weights is None:
            loss = self.loss_fn(y_pred, y)
        else:
            loss = self.naive_loss(y_pred, y)

        #Using the gradient clip to protect from gradient exploration
        if self.config["gradient_clip"]:
            nn.utils.clip_grad_norm_(self.network.parameters(), 1)

        l1_regularization = torch.tensor(0., device=device)
        for name, param in self.network.named_parameters(): 
            if 'Attention_layer.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention_layer.W2' in name:
                l1_regularization += torch.norm(param, p=1)

        loss += l1_regularization*0.001

        loss += self._attention_regularizer(torch.transpose(self.network.att, 1, 2))
        self.log("train_loss", loss, on_epoch = True, on_step = True)

        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)
        
        self.log('train categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('train categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('train binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)
 
        return loss
    def categorical_accuracy_strict(self, y_pred, y_true):
    # Find the index of the maximum value in each row (i.e., the predicted class)
        y_pred_class = torch.round(y_pred)
        com = y_pred_class == y_true
        correct = com.all(dim=1).sum()
        sample_num = y_true.size(0)
        accuracy = correct / sample_num
        return accuracy
    def validation_step(self, batch):

        x, mask, RNA_type, y=  batch["embedding"], batch["attention_mask"], batch["RNA_type"], batch["labels"]
        y = y.to(torch.float32)
        y_pred = self.forward(x, mask, RNA_type)

        categorical_accuracy = self.categorical_accuracy(y_pred, y)
        categorical_accuracy_strict = self.categorical_accuracy_strict(y_pred, y)
        binary_accuracy = self.binary_accuracy(y_pred, y)

        self.log('val categorical_accuracy', categorical_accuracy, on_step = True, on_epoch=True, prog_bar = True)
        self.log('val categorical_accuracy_strict', categorical_accuracy_strict, on_step = True, on_epoch=True, prog_bar = True)
        self.log('val binary_accuracy', binary_accuracy, on_step = True, on_epoch=True, prog_bar = True)

        if self.class_weights is None:
            loss = self.loss_fn(y_pred, y)
        else:
            loss = self.naive_loss(y_pred, y)
        l1_regularization = torch.tensor(0., device=device)
        for name, param in self.network.named_parameters(): 
            if 'Attention_layer.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention_layer.W2' in name:
                l1_regularization += torch.norm(param, p=1)

        loss += l1_regularization*0.001
        loss += self._attention_regularizer(torch.transpose(self.network.att, 1, 2))
           

        self.log("val_loss", loss, on_epoch = True, on_step = True)

        return {"categorical_accuracy": categorical_accuracy, "categorical_accuracy_strict":categorical_accuracy_strict,
                "binary_accuracy": binary_accuracy}
    def print_init_parameters(self):
        init_params = inspect.signature(self.__init__).parameters
        param_names = [param for param in init_params if param != 'self']
        for param_name in param_names:
            param_value = getattr(self, param_name)
            print(f"{param_name}: {param_value}")



class MyTrainer:
    def __init__(self, model, config):
        self.config = config
        self.plmodel = LightningModel(model, class_weights = self.config.class_weights)
        self.wandb_logger = WandbLogger(log_model = "all")
        self.ckpt_path = path_join(self.config.output_dir, "checkpoints")
        self.pred = None
        self.test = None

    def train(self, train_loader, val_loader):
        samples = next(iter(train_loader))
        embed = samples["embedding"][:2].shape
        mask = samples["attention_mask"][:2].shape
        RNA_types = samples["RNA_type"][:2].shape
        summary(self.plmodel, input_size = [embed, mask, RNA_types], device = device)
        trainer = Trainer(accelerator = self.config.accelerator, strategy = self.config.strategy, max_epochs = self.config.max_epochs, devices = gpus, precision = self.config.precison, num_nodes = self.config.num_nodes, 
                logger = self.wandb_logger,
                log_every_n_steps = self.config.log_every_n_steps,
                callbacks = self.make_callback())
        trainer.fit(self.plmodel, train_loader, val_loader)

    def make_callback(self):

        callbacks = [
            ModelCheckpoint(dirpath = self.ckpt_path, filename = "checkpoints_best", save_top_k = 1, verbose = True, mode = "min", monitor = "val_loss"),
            EarlyStopping(monitor = "val_loss", min_delta = 0.00, patience = self.config.patience, verbose = True, mode = "min")
            ]
        return callbacks
    
    def load_checkpoint(self):
       
        checkpoint = torch.load(path_join(self.ckpt_path, "checkpoints_best.ckpt"))
        model_state = checkpoint['state_dict']
        self.plmodel.load_state_dict(model_state)
        return self.plmodel #model with new parameters
    
    def test(self, test_loader):

        self.plmodel = self.load_checkpoint().network
        self.plmodel.eval()
        self.get_metrics_all(test_loader)


    def get_metrics_all(self, dataloader_test, model):
        all_y_pred = []
        all_y_test = []
        all_RNA_types = []
        for idx, batch in enumerate(dataloader_test):
            X_test, X_mask, RNA_type, y_test = batch["embedding"], batch["attention_mask"], batch["RNA_type"], batch["labels"]
            y_pred = self.plmodel.forward(X_test, X_mask, RNA_type)
            y_pred = y_pred.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            RNA_type = RNA_type.detach().cpu().numpy()
            all_y_pred.append(y_pred)
            all_y_test.append(y_test)
            all_RNA_types.append(RNA_type)
        y_test = np.concatenate(all_y_test, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        RNA_types = np.concatenate(all_RNA_types, axis=0)
        RNA_types = [self.config.RNA_order[i] for i in RNA_types]
        #split the data with RNA types
        RNA_libs = ["lncRNA", "miRNA", "snRNA", "snoRNA", "mRNA"]
        dfs = []
        for RNA in RNA_libs:
            idx = np.where(np.isin(RNA_types, [RNA]))[0]
            if RNA == "lncRNA":
                idx = np.where(np.isin(RNA_types, ["lncRNA", "lincRNA"]))[0]
            
            try:
                y_pred_sub = y_pred[idx]
                y_test_sub = y_test[idx]
                best_t = self.find_best_t(y_pred_sub, y_test_sub)
                df = self.cal_metrics(y_test[idx], y_pred[idx], RNA, best_t)
                dfs.append(df)
                
            except:
                print("empty:", RNA)
        all_df = pd.concat(dfs, axis=0, ignore_index=True)

        all_df.to_csv(path_join(self.config.ourput_dir, "all_metrics.csv"), index = False)
        return all_df
    def cal_metrics(self, y_test : np.ndarray, y_pred : np.ndarray, RNA : str, best_t : Dict):
        num_classes = y_test.shape[1]
        metrics_data = []
        mcc_sum = 0
        for c in range(num_classes):  # calculate one by one
            t = best_t[c]
            try:
                average_precision_c = average_precision_score(y_test[:, c], y_pred[:, c])
                roc_auc_c = roc_auc_score(y_test[:, c], y_pred[:, c])
            except ValueError:
                average_precision_c = np.nan
                roc_auc_c = np.nan

            y_pred_bi = np.where(y_pred[:, c] > t, 1, 0)
            mcc_c = matthews_corrcoef(y_test[:, c], y_pred_bi)
            mcc_sum += mcc_c
            f1_score_c = f1_score(y_test[:, c], y_pred_bi)
            precision_c = precision_score(y_test[:, c], y_pred_bi)
            recall_c = recall_score(y_test[:, c], y_pred_bi)
            acc_c = accuracy_score(y_test[:, c], y_pred_bi)

            metrics_data.append({
                'RNA': RNA,
                'Compartment': f'Compartment_{c}',  # You might want to replace this with the actual compartment name
                'AveragePrecision': average_precision_c,
                'ROCAUC': roc_auc_c,
                'MatthewsCorrCoef': mcc_c,
                'F1Score': f1_score_c,
                'Precision': precision_c,
                'Recall': recall_c,
                'Accuracy': acc_c
            })

        metrics_data.append({
            'RNA': RNA,
            'Compartment': 'micro',
            'AveragePrecision': average_precision_score(y_test, y_pred, average='micro'),
            'ROCAUC': roc_auc_score(y_test, y_pred, average='micro'),
            'MatthewsCorrCoef': matthews_corrcoef(y_test.ravel(), y_pred_bi.ravel()),
            'F1Score': f1_score(y_test.ravel(), y_pred_bi.ravel(), average='micro'),
            'Precision': precision_score(y_test.ravel(), y_pred_bi.ravel(), average='micro'),
            'Recall': recall_score(y_test.ravel(), y_pred_bi.ravel(), average='micro'),
            'Accuracy': accuracy_score(y_test.ravel(), y_pred_bi.ravel())
        })

        metrics_data.append({
            'RNA': RNA,
            'Compartment': 'macro',
            'AveragePrecision': average_precision_score(y_test, y_pred, average='macro'),
            'ROCAUC': roc_auc_score(y_test, y_pred, average='macro'),
            'MatthewsCorrCoef': mcc_sum / num_classes,
            'F1Score': f1_score(y_test, y_pred_bi, average='macro'),
            'Precision': precision_score(y_test, y_pred_bi, average='macro'),
            'Recall': recall_score(y_test, y_pred_bi, average='macro'),
            'Accuracy': accuracy_score(y_test, y_pred_bi, average='macro')
        })

        df = pd.DataFrame(metrics_data)
        return df

    def get_mcc(self, t):
        mcc = matthews_corrcoef(self.test, [1 if i>t else 0 for i in self.pred])
        return mcc
    def find_best_t(self, y_pred : List, y_test : List) -> None:
        print("Calculating the best threshold for each target ...")
        num_classes = y_test.shape[1]
        ts = np.linspace(0, 1, 50)
        best_t = {}
        for c in range(num_classes):#calculate one by one
            self.pred = y_pred[:,c]
            self.test = y_test[:,c]
            max_idx = np.argmax(np.array(map(self.get_mcc, ts)))
            max_t = ts[max_idx]
            best_t[c] = max_t
        return best_t



        


