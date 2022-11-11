
import torch.optim as optim
import numpy as np

def Adam(model,learning_rate ):
    return optim.Adam(model.parameters(), lr= learning_rate)

def SGD(model,learning_rate,momentum=0,dampening=0,weight_decay=0):
    return optim.SGD(model.parameters(), lr= learning_rate,momentum=momentum,dampening=dampening,weight_decay=weight_decay)

def adjust_learning_rate(optimizer, epoch,learning_rate, lr_steps=[40, 55, 65, 70, 200, 250]):
    lr_new = learning_rate * (0.1 ** (sum(epoch >= np.array(lr_steps))))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_new