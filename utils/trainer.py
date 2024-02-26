
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
import glob
import torch.nn.functional as F

#make the code reproducible 
# Set random seed for NumPy
np.random.seed(42)

# Set random seed for PyTorch
torch.manual_seed(42)

# If using GPU, set random seed for CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
# gpus = torch.cuda.device_count()
gpus = 1
path_join = lambda *path: os.path.abspath(os.path.join(*path))



class MultiTaskLossWrapper(nn.Module):
    def __init__(self, num_task):
        super(MultiTaskLossWrapper, self).__init__()
        self.num_task = num_task
        self.log_vars = nn.Parameter(torch.zeros((num_task)))
    def binary_cross_entropy(self, x, y):
        epsilon = 1e-4
        x = torch.clamp(x, epsilon, 1 - epsilon)
        loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
        return torch.mean(loss)
    def forward(self, y_pred,targets):
        loss_output = torch.zeros(1).to(device = device)
        for i in range(self.num_task):
            out = torch.exp(-self.log_vars[i])*self.binary_cross_entropy(y_pred[:,i],targets[:,i]) + self.log_vars[i]
            loss_output += out
            print("loss_output", loss_output)
        loss = loss_output/self.num_task
        print("loss", loss)

        return loss

class LightningModel(pl.LightningModule):
    def __init__(self, model, config = None, 
                 weight_dict = None, flt_dict = None):
        super(LightningModel, self).__init__()
        #only used to extract the RNA types
        self.network = model.to(device)
        self.config = config
        self.epoch_start_time = None
        self.class_weights = weight_dict
        self.flt_dict = flt_dict
        self.learnable_loss = MultiTaskLossWrapper(self.config.nb_classes)
        self.learnable_loss.to(device)
        self.loss_fn = nn.BCELoss()
        print("pl model has be loaded")
    def binary_cross_entropy(self, x, y,focal=True):
        alpha = 0.7
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
    
    def hier_loss(self, y_pred, y_true, RNA_type, ohem=False, focal=False):
        
        #get number of sample per batch
        num_examples = y_true.shape[0]

        #initialize the loss for all samples
        loss_output = torch.zeros(num_examples).to(device = device)
        #get new index of calculated rna -> Tensor[int]
        RNA_tensor = torch.tensor([self.config.rna_ref.index(self.config.RNA_order[int(i)]) for i in RNA_type], device=device)

        for rna_int in torch.unique(RNA_tensor):
            #get rna string
            rna_str = self.config.rna_ref[int(rna_int)]
            rna_idx = torch.where(RNA_tensor == rna_int)[0]
            #only calculate the weight loss with enough sample size
            class_n = len(self.flt_dict[rna_str])
            for i,c in enumerate(self.flt_dict[rna_str]):
                out = self.binary_cross_entropy(y_pred[rna_idx,c],y_true[rna_idx,c],focal)*(1/class_n)
                loss_output[rna_idx] += out

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
        print("start forward")
        embed = embed.to(device)
        print("embed")
        mask = mask.to(device)
        if RNA_type != None:
            RNA_type = RNA_type.to(device)
        # print("string:",str(self.network))
        pred = self.network(embed, mask, RNA_type)
        print("after prediction")
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

    def training_step(self, batch, batch_idx, **kwargs):

        x, mask, RNA_type, y = batch["embedding"], batch["attention_mask"], batch["RNA_type"], batch["labels"]
        y = y.to(torch.float32)
        y_pred = self.forward(x, mask, RNA_type)

        if self.class_weights is None:
            loss = self.loss_fn(y_pred, y)
        else:
            loss = self.hier_weight_loss(y_pred, y, RNA_type)

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
        if not self.config.lora:
            loss += self._attention_regularizer(torch.transpose(self.network.att_weight, 1, 2))
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
    def validation_step(self, batch, batch_idx, **kwargs):

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
            loss = self.hier_weight_loss(y_pred, y, RNA_type)
        l1_regularization = torch.tensor(0., device=device)
        for name, param in self.network.named_parameters(): 
            if 'Attention_layer.W1' in name:
                l1_regularization += torch.norm(param, p=1)
            elif 'Attention_layer.W2' in name:
                l1_regularization += torch.norm(param, p=1)

        loss += l1_regularization*0.001
        if not self.config.lora:
            loss += self._attention_regularizer(torch.transpose(self.network.att_weight, 1, 2))
           

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
    def __init__(self, model, config, weight_dict = None, flt_dict = None):
        self.config = config
        self.plmodel = LightningModel(model, config = self.config, 
                                      weight_dict = weight_dict, 
                                      flt_dict = flt_dict)
        self.wandb_logger = WandbLogger(log_model = "all")
        self.ckpt_path = path_join(self.config.output_dir, "checkpoints")
        self.pred = None
        self.test_y = None
        self.plmodel.to(device)
        


    def train(self, train_loader, val_loader):
    
        samples = next(iter(train_loader))

        # embed = samples["embedding"][:2].shape
        # mask = samples["attention_mask"][:2].shape
        # RNA_types = samples["RNA_type"][:2].shape
        # print(embed,mask,RNA_types)
        print("embedding:", samples["embedding"][:2].shape)
        print("mask:", samples["attention_mask"][:2].shape)
        print("RNA_type", samples["RNA_type"][:2].shape)
        embed = samples["embedding"][:2].to(device)
        mask = samples["attention_mask"][:2].to(device)
        RNA_types = samples["RNA_type"][:2].to(device)
        # summary_str = summary(self.plmodel, input_size = [embed, mask, RNA_types], device = device)

        #saving the model summary
        # summary_file = "model_summary.txt"
        # Write the summary to a file
        # with open(path_join(self.config.output_dir, summary_file), "w") as file1:
        #     file1.write(summary_str)
        #also save the model details 
        detail_str = str(self.plmodel.network)
        detail_file = "model_details.txt"
        with open(path_join(self.config.output_dir, detail_file), "w") as file2:
            file2.write(detail_str)


        trainer = Trainer(accelerator = self.config.accelerator, strategy = self.config.strategy, max_epochs = self.config.max_epochs, devices = gpus, precision = self.config.precison, num_nodes = self.config.num_nodes, 
                logger = self.wandb_logger,
                log_every_n_steps = self.config.log_every_n_steps,
                callbacks = self.make_callback())
        print("after trainer")
        trainer.fit(self.plmodel, train_loader, val_loader)
        print("after fitting")
        # write model summary
        with open(path_join(self.config.output_dir, "model.summary.txt"), "w") as f:
            print(str(self.plmodel), file=f)


    def make_callback(self):

        callbacks = [
            ModelCheckpoint(dirpath = self.ckpt_path, filename = "checkpoints_best", save_top_k = 1, verbose = True, mode = "min", monitor = "val_loss"),
            EarlyStopping(monitor = "val_loss", min_delta = 0.00, patience = self.config.patience, verbose = True, mode = "min")
            ]
        return callbacks
    
    def load_checkpoint(self):
        checkpoint_orig = glob.glob(path_join(self.ckpt_path, "checkpoints_best.ckpt"))
        # print("checkpoint_orig", checkpoint_orig)
        checkpoint_topup = glob.glob(path_join(self.ckpt_path, "checkpoints_best-v*.ckpt"))
        # print("checkpoint_topup", checkpoint_topup)
        if not checkpoint_orig:
            print("No checkpoint files found.")
        else:
            if checkpoint_topup:
                latest_checkpoint = max(checkpoint_topup, key=lambda x: int(x.split('-v')[-1].split('.')[0]))
                checkpoint = torch.load(latest_checkpoint, map_location = device)
            else:
                checkpoint = torch.load(checkpoint_orig[0], map_location = device)
        # checkpoint = torch.load(path_join(self.ckpt_path, "checkpoints_best-v24.ckpt"), map_location = device)
        model_state = checkpoint['state_dict']
        self.plmodel.load_state_dict(model_state)
        return self.plmodel #model with new parameters
    
    def test(self, test_loader, flt_dict):

        #only running the evaluation in 1 gpu

        # if torch.distributed.get_rank() not in [0, -1]:
        #     torch.distributed.barrier()

        print("evaluating only on one gpu")
        torch.distributed.destroy_process_group()
        self.plmodel = self.load_checkpoint()
        self.plmodel = self.plmodel.to(device)
        self.plmodel.eval()
        
        self.get_metrics_all(test_loader, flt_dict)

        # if torch.distributed.get_rank() == 0:
        #     torch.distributed.barrier()
        


    def get_metrics_all(self, dataloader_test, flt_dict):
        all_y_pred = []
        all_y_test = []
        all_RNA_types = []
        with torch.no_grad():
            for batch in dataloader_test:
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
        RNA_types = [self.config.RNA_order[int(i)] for i in RNA_types]
        #split the data with RNA types
        RNA_libs = ["lncRNA", "miRNA", "snRNA", "snoRNA", "mRNA"]
        dfs = []
        for RNA in RNA_libs:
            idx = np.where(np.isin(RNA_types, [RNA]))[0]
            if RNA == "lncRNA":
                idx = np.where(np.isin(RNA_types, ["lncRNA", "lincRNA"]))[0]
            # try:
            y_pred_sub = y_pred[idx]
            y_test_sub = y_test[idx]
            best_t = self.find_best_t(y_pred_sub, y_test_sub)
            flt_targets = flt_dict[RNA]
            df = self.cal_metrics(y_test[idx], y_pred[idx], RNA, best_t, flt_targets)
            dfs.append(df)
        all_df = pd.concat(dfs, axis=0, ignore_index=True)

        all_df.to_csv(path_join(self.config.output_dir, "all_metrics.csv"), index = False, na_rep="NAN")
        return all_df
    def cal_metrics(self, y_test : np.ndarray, y_pred : np.ndarray, RNA : str, best_t : Dict, flt_targets : List):
        num_classes = y_test.shape[1]
        metrics_data = []
        mcc_sum = 0
        acc_sum = 0
        # unused_class = []
        # print("calculating metrics:", y_test, y_pred)
        valid_tg = np.where(y_test.sum(axis = 0) > 0)[0]
        for c in range(num_classes):  # calculate one by one
            t = best_t[c]
            cn = self.config.compartments[c]
            if c in valid_tg:
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
                acc_sum += acc_c

                metrics_data.append({
                    'RNA': RNA,
                    'Compartment': f'{cn}',  # You might want to replace this with the actual compartment name
                    'Count': np.sum(y_test[:, c]),
                    'AveragePrecision': average_precision_c,
                    'ROCAUC': roc_auc_c,
                    'MatthewsCorrCoef': mcc_c,
                    'F1Score': f1_score_c,
                    'Precision': precision_c,
                    'Recall': recall_c,
                    'Accuracy': acc_c
                })
            else:
                # unused_class.append(c)
                metrics_data.append({
                    'RNA': RNA,
                    'Compartment': f'{cn}',  # You might want to replace this with the actual compartment name
                    'Count': np.nan,
                    'AveragePrecision': np.nan,
                    'ROCAUC': np.nan,
                    'MatthewsCorrCoef': np.nan,
                    'F1Score': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'Accuracy': np.nan
                })
        #for micro and macro
        if len(flt_targets) > 0:
            #exclude the cytoplasm when calculating micro and macro
            flt_targets = [i for i in flt_targets if i != self.config.compartments.index("Cytoplasm")]
            #keep target pass the thredholds
            y_test_flt = y_test[:,flt_targets]
            y_pred_flt = y_pred[:,flt_targets]
            # y_pred_bi_ac = np.where(y_pred_flt > t, 1, 0)
            y_pred_bi_ac = np.concatenate([np.where(y_pred[:, c] > best_t.get(c), 1, 0)[:, None] for c in flt_targets], axis=1)


            auprc_micro = average_precision_score(y_test_flt, y_pred_flt, average='micro')
            auprc_macro = average_precision_score(y_test_flt, y_pred_flt, average='macro')
            try:
                auroc_micro = roc_auc_score(y_test_flt, y_pred_flt, average='micro')
                auroc_macro = roc_auc_score(y_test_flt, y_pred_flt, average='macro')
            except ValueError:
                auroc_micro = np.nan
                auroc_macro = np.nan

            # print("y_test_flt", y_test_flt.shape, y_test_flt)
            # print("y_pred_bi_ac", y_pred_bi_ac.shape, y_pred_bi_ac)
            metrics_data.append({
                'RNA': RNA,
                'Compartment': 'micro',
                'Count': np.sum(y_test_flt.ravel()),
                'AveragePrecision': auprc_micro,
                'ROCAUC': auroc_micro,
                'MatthewsCorrCoef': matthews_corrcoef(y_test_flt.ravel(), y_pred_bi_ac.ravel()),
                'F1Score': f1_score(y_test_flt.ravel(), y_pred_bi_ac.ravel(), average='micro'),
                'Precision': precision_score(y_test_flt.ravel(), y_pred_bi_ac.ravel(), average='micro'),
                'Recall': recall_score(y_test_flt.ravel(), y_pred_bi_ac.ravel(), average='micro'),
                'Accuracy': accuracy_score(y_test_flt.ravel(), y_pred_bi_ac.ravel())
            })

            metrics_data.append({
                'RNA': RNA,
                'Compartment': 'macro',
                'Count': np.sum(y_test_flt.ravel()),
                'AveragePrecision': auprc_macro,
                'ROCAUC': auroc_macro,
                'MatthewsCorrCoef': mcc_sum / len(flt_targets),
                'F1Score': f1_score(y_test_flt, y_pred_bi_ac, average='macro'),
                'Precision': precision_score(y_test_flt, y_pred_bi_ac, average='macro'),
                'Recall': recall_score(y_test_flt, y_pred_bi_ac, average='macro'),
                'Accuracy': acc_sum / len(flt_targets)
            })
        else:
            metrics_data.append({
                'RNA': RNA,
                'Compartment': 'micro',
                'Count': 0,
                'AveragePrecision': np.nan,
                'ROCAUC': np.nan,
                'MatthewsCorrCoef': np.nan,
                'F1Score': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'Accuracy': np.nan
            })

            metrics_data.append({
                'RNA': RNA,
                'Compartment': 'macro',
                'Count': 0,
                'AveragePrecision': np.nan,
                'ROCAUC': np.nan,
                'MatthewsCorrCoef': np.nan,
                'F1Score': np.nan,
                'Precision': np.nan,
                'Recall': np.nan,
                'Accuracy': np.nan
            })


        df = pd.DataFrame(metrics_data)
        return df

    def get_mcc(self, t):
        mcc = matthews_corrcoef(self.test_y, [1 if i>t else 0 for i in self.pred])
        return mcc
    def find_best_t(self, y_pred : List, y_test : List) -> Dict:
        print("Calculating the best threshold for each target ...")
        num_classes = y_test.shape[1]
        ts = np.linspace(0, 1, 50)
        best_t = {}
        for c in range(num_classes):#calculate one by one
            self.pred = y_pred[:,c]
            self.test_y = y_test[:,c]
            mcc_t = list(map(self.get_mcc, ts))
            max_idx = mcc_t.index(max(mcc_t))
            max_t = ts[max_idx]
            best_t[c] = max_t
        return best_t



        



