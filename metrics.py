from torchmetrics import Accuracy


def accuracy(output, labels,top_k=1):

    acc = Accuracy(top_k=top_k)
    
    return acc(output,labels)