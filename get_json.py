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