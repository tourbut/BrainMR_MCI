import torch

if __name__ == "__main__":
    print('hello world')

    device = torch.device('mps')
    print(f" - device : {device}")
    sample = torch.Tensor([[10, 20, 30], [30, 20, 10]])
    print(f" - cpu tensor : ")
    print(sample)
    sample = sample.to(device)
    print(f" - gpu tensor : ")
    print(sample)

    