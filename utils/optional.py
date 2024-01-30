from typing import List, Tuple, Dict, Union, Sequence
import numpy as np
import os
import json
path_join = lambda *path: os.path.abspath(os.path.join(*path))
class Weights:
    def __init__(self, nb_classes, sample_t):
        self.nb_classes = nb_classes
        self.sample_t = sample_t 
    

    def cal_loss_weight(self, y : np.ndarray,  filtered_labels : List, beta : float = 0.99999):
        
        labels_dict = dict(zip(range(len(filtered_labels)),[sum(y[:,i]) for i in filtered_labels]))
        keys = labels_dict.keys()
        class_weight = dict()

        # Class-Balanced Loss Based on Effective Number of Samples
        for key in keys:
            effective_num = 1.0 - beta**labels_dict[key]
            weights = (1.0 - beta) / effective_num
            class_weight[key] = weights

        weights_sum = sum(class_weight.values())

        # normalizing weights
        for key in keys:
            class_weight[key] = class_weight[key] / weights_sum * 10

        return class_weight 

    def get_weight(self, RNA_list: List[int], labels: List[str], RNA_order: List, save_dir : str ) -> Dict:
        weight_dict = {}
        flt_dict = {}
        unique_rnas = list(np.sort(np.unique(RNA_list)))
        labels_array = np.array(labels)
        # import pdb; pdb.set_trace()
        all_labels_int = np.array([list(map(int, label)) for label in labels])
        pos_weight = 1/(np.sum(all_labels_int, axis = 0)/all_labels_int.shape[0])
        #lower down the mito
        pos_weight[-1] = pos_weight[-2]*2
        print("pos weight:", pos_weight)

        for rna in unique_rnas:
            rna = int(rna)
            # import pdb; pdb.set_trace()
            rna_str = RNA_order[rna]
            if rna_str != "lincRNA" or rna_str != "lincRNA":
                # Filter indices of the current RNA in RNA_list
                idxs = np.where(np.array(RNA_list) == rna)[0]

                # Extract labels for the current RNA
                rna_labels = labels_array[idxs]

                # Convert labels to a 2D array of integers
                labels_int = np.array([list(map(int, label)) for label in rna_labels])
                total = labels_int.sum(axis=0)
                filtered_col = np.where(total > self.sample_t)[0]
                #initialize the weight
                class_weights = np.zeros(labels_int.shape[1])
                #only calculate the weight of the picked labels
                class_weights_flt = list(self.cal_loss_weight(labels_int, filtered_labels = filtered_col).values())
                #keep others 0 and selected labels as weights
                class_weights[filtered_col] = class_weights_flt
                # Store the result in the dictionary
                weight_dict[rna_str] = list(class_weights)
                flt_dict[rna_str] = [int(i) for i in filtered_col]
                
            else:
                # Filter indices of the current RNA in RNA_list
                lnc = [RNA_order.index("lncRNA"), RNA_order.index("lincRNA")]
                idxs = np.where(np.isin(RNA_list, lnc))[0]
                
                # Extract labels for the current RNA
                rna_labels = labels_array[idxs]
                # Convert labels to a 2D array of integers
                labels_int = np.array([list(map(int, label)) for label in rna_labels])
                total = labels_int.sum(axis=0)
                # import pdb; pdb.set_trace()
                filtered_col = np.where(total > self.sample_t)[0]
                #calculate weight according to the fifltered samples
                class_weights = np.zeros(labels_int.shape[1])
                class_weights_flt = list(self.cal_loss_weight(labels_int, filtered_labels = filtered_col).values())
                # Store the result in the dictionary
                class_weights[filtered_col] = class_weights_flt
                weight_dict["lncRNA"] = list(class_weights)
                flt_dict["lncRNA"] = [int(i) for i in filtered_col]
        if save_dir:
            with open(path_join(save_dir, "weights.json"), "w") as json_file:
                print("weight_dict", weight_dict)
                json.dump(weight_dict, json_file)
            with open(path_join(save_dir, "filtered_targets.json"), "w") as json_file:
                print("flt_dict",flt_dict)
                json.dump(flt_dict, json_file)
        
        
        return weight_dict, flt_dict, pos_weight

