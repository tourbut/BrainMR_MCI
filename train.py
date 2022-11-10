from torch.autograd import Variable
import time

from metrics import accuracy

from utils import *

def train(device, epoch, data_loader, model, criterion, optimizer, 
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train(True)
    losses = AverageMeter(name='losses')
    accuracies = AverageMeter(name='accuracies')

    for i, (inputs, targets) in enumerate(data_loader):

        inputs = Variable(inputs).to(device)
        targets = Variable(targets).to(device)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        acc = accuracy(outputs.data, targets.data,device=device)

        accuracies.update(acc, inputs.size(0))
        losses.update(loss.data, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'acc': accuracies.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Loss : {loss.avg:.4f}\t'
                  'Acc : {acc.avg:.5f}\t'.format(
                      epoch,
                      i,
                      len(data_loader),
                      loss=losses,
                      acc=accuracies,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'acc': accuracies.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })
    return model, losses.avg.item(), accuracies.avg.item()
