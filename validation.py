from torch.autograd import Variable
import time

from metrics import accuracy

from utils import *

def validation(device, epoch, data_loader, model, criterion, logger):
    print('valid at epoch {}'.format(epoch))

    model.eval()
    losses = AverageMeter(name='losses')
    accuracies = AverageMeter(name='accuracies')

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):

            inputs = Variable(inputs).to(device)
            targets = Variable(targets).to(device)
                
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs.data, targets.data,device=device)

            losses.update(loss.data, inputs.size(0))
            accuracies.update(acc, inputs.size(0))


            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t '
                    'Loss : {loss.avg:.4f}\t'
                    'Acc : {acc.avg:.5f}\t'.format(
                        epoch,
                        i,
                        len(data_loader),
                        loss=losses,
                        acc=accuracies))

    logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'acc': accuracies.avg.item()
    })
    print('Epoch: [{0}]\t '
        'Loss : {loss.avg:.4f}\t'
        'Acc : {acc.avg:.5f}\t'.format(
            epoch,
            loss=losses,
            acc=accuracies))
    
    return losses.avg.item(), accuracies.avg.item()
