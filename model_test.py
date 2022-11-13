from torch.autograd import Variable
import torch

from metrics import accuracy

import utils

import os
import datetime

from torchmetrics.functional.classification import multiclass_auroc
from torchmetrics.classification import MulticlassConfusionMatrix

def test(device, data_loader, model, criterion, config,age_onoff = True):
    print('test')

    log_path = config['log_path']
    log_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = config['model']['model_name']
    model_depth = config['model']['model_depth']
    store_name = model_name + str(model_depth)
    store_name = store_name+'_'+log_date

    logger = utils.Logger(os.path.join(log_path, store_name+'_train.log'),['epoch', 'loss','acc', 'lr'])
    
    model.eval()
    losses = utils.AverageMeter(name='losses')
    accuracies = utils.AverageMeter(name='accuracies')

    pred = []
    labels = []
    with torch.no_grad():
        for i, (inputs, input_age, targets) in enumerate(data_loader):

            inputs = Variable(inputs).to(device)
            targets = Variable(targets).to(device)
                
            if age_onoff == True:
                outputs = model(inputs,input_age)
            else:
                outputs = model(inputs)

            pred.append(outputs[0].data.cpu())
            labels.append(targets[0].data.cpu())    

            loss = criterion(outputs, targets)
            acc = accuracy(outputs.data, targets.data,device=device)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            
    pred = torch.stack(pred)
    labels =  torch.stack(labels)
    
    metric = MulticlassConfusionMatrix(num_classes=3)

    ConfusionMatrix = metric(outputs, labels)

    auroc = multiclass_auroc(pred, labels, num_classes=3, average=None, thresholds=None)

    logger.log({
        'loss': losses.avg.item(),
        'acc': accuracies.avg.item(),
        'ConfusionMatrix' : ConfusionMatrix,
        'auroc' : auroc
    })


    print('Loss : {loss.avg:.4f}\t Acc : {acc.avg:.5f}\t'.format(loss=losses, acc=accuracies))
    
    return losses.avg.item(), accuracies.avg.item(),ConfusionMatrix,auroc
