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




