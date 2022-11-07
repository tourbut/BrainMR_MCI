from torch.autograd import Variable
import time

from metrics import accuracy

from utils import *

def test(device, data_loader, model, criterion, logger):
    print('test')

    model.eval()
    losses = AverageMeter(name='losses')
    accuracies = AverageMeter(name='accuracies')

    for i, (inputs, targets) in enumerate(data_loader):

        with torch.no_grad():
            inputs = Variable(inputs).to(device)
            targets = Variable(targets).to(device)
            
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, inputs.size(0))
        acc = accuracy(outputs.data, targets.data,device=device)
        accuracies.update(acc, inputs.size(0))


    print('Loss {loss.avg:.4f}, acc {acc.avg:.5f}'.format(loss=losses, acc=accuracies))

    logger.log({
        'loss': losses.avg.item(),
        'acc': accuracies.avg.item()
    })
    
    return losses.avg.item(), accuracies.avg.item()
