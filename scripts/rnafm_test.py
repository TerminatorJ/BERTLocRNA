import torch
import fm

# Load RNA-FM model
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Prepare data
data = [
    ("RNA1", "GGGUGCGAUCAUACCAGCACUAAUGCCCUCCUGGGAAGUCCUCGUGUUGCACCCCU")]
#     ("RNA1", "GGGUGUCGCUCAGUUGGUAGAGUGCUUGCCUGGCAUGCAAGAAACCUUGGUUCAAUCCCCAGCACUGCA"),
#     ("RNA1", "CGAUUCNCGUUCCC--CCGCCUCCA"),
# ]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
print("batch_tokens", batch_tokens.shape)
# Extract embeddings (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12])
token_embeddings = results["representations"][12]
print("token_embeddings", token_embeddings.shape)