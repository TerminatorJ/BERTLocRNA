from typing import List, Tuple, Dict, Union
import numpy as np

def filter_sequences(seqs, effective_length):
    truncated = {}
    seq_modified = []
    for idx,seq in enumerate(seqs):
        if len(seq) > effective_length:
            seq_modified.append(seq[:effective_length])
            truncated[str(idx)] = seq[effective_length:]
        else:
            seq_modified.append(seq)

    return seq_modified, truncated

def merge_input(id_all, id_truncated, truncated_dict : Dict, key) -> np.ndarray:
    count = 0
    final_list = []

    for idx, input_id in enumerate(id_all[key]):
        if str(idx) in truncated_dict.keys():

            new_id = input_id + id_truncated[key][count]
            final_list.append(new_id)
            count+=1
        else:
            final_list.append(input_id)

    return final_list


