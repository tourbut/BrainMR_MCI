from torchmetrics import Accuracy
from sklearn.metrics import roc_curve, auc, roc_auc_score


def accuracy(output, labels,top_k=1,device='cpu'):
    acc = Accuracy(top_k=top_k).to(device)
    return acc(output,labels)