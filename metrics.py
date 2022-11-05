from torchmetrics import Accuracy


def accuracy(output, labels,top_k=1,device='cpu'):

    acc = Accuracy(top_k=top_k).to(device)
    
    return acc(output,labels)