import json
import csv


def load_txt(txt_dir, txt_name):
    List = []
    with open(txt_dir + txt_name, 'r') as f:
        for line in f:
            List.append(line.strip('\n').replace('.nii', '.npy'))
    return List

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    labels = [0 if a[1]=='CN' else 1 for a in your_list[1:]]
    return filenames, labels

