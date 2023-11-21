import json
from collections import OrderedDict

#generate manually token id
my_ordered_dict = OrderedDict([
    ('UNK', [0, 0, 0, 0]), 
    ('A', [1, 0, 0, 0]), 
    ('C', [0, 1, 0, 0]), 
    ('G', [0, 0, 1, 0]), 
    ('T', [0, 0, 0, 1]), 
    ('N', [0.25, 0.25, 0.25, 0.25]), 
    ('Y RNA', [0, 0, 0, 0]), 
    ('lincRNA', [0, 0, 0, 0]), 
    ('lncRNA', [0, 0, 0, 0]), 
    ('mRNA', [0, 0, 0, 0]), 
    ('miRNA', [0, 0, 0, 0]), 
    ('ncRNA', [0, 0, 0, 0]), 
    ('pseudo', [0, 0, 0, 0]), 
    ('rRNA', [0, 0, 0, 0]), 
    ('scRNA', [0, 0, 0, 0]), 
    ('scaRNA', [0, 0, 0, 0]), 
    ('snRNA', [0, 0, 0, 0]), 
    ('snoRNA', [0, 0, 0, 0]), 
    ('vRNA', [0, 0, 0, 0])])


# Specify the path where you want to save the JSON file
json_file_path = './vocabulary.json'

# Write the OrderedDict to the JSON file
with open(json_file_path, 'w') as json_file:
    json.dump(my_ordered_dict, json_file, indent=0) 

with open(json_file_path, 'r') as json_file:
    print(json.load(json_file))

#the model configuration and json file generation
model_ordered_dict = OrderedDict([
    ('NT', {"pretrained_model_name_or_path" : "InstaDeepAI/nucleotide-transformer-2.5b-multi-species", "trust_remote_code" : False}), 
    ('DNABERT2', {"pretrained_model_name_or_path" : "zhihan1996/DNABERT-2-117M", "trust_remote_code" : True})])

model_json_file = './model.json'
with open(model_json_file, 'w') as json_file:
    json.dump(model_ordered_dict, json_file, indent=0) 
with open(model_json_file, 'r') as json_file:
    print(json.load(json_file))