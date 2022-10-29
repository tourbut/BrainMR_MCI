import torch
from utils import read_json
import dataloader 

def load_config():
    config = read_json('./config.json')
    return config

if __name__ == "__main__":
    config = load_config()

    device_mode = config['device']
    print('PROJECT_NAME: ',config['PROJECT_NAME'],'\n')

    device = torch.device(device_mode)
    print(f" - device : {device}")

    sample = torch.Tensor([[10, 20, 30], [30, 20, 10]])
    print(f" - cpu tensor : ")

    print(sample)
    sample = sample.to(device)
    print(f" - gpu tensor : ")
    
    print(sample)

