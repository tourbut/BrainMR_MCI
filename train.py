from torch.autograd import Variable
import time

from metrics import accuracy

from utils import *

def train(device, epoch, data_loader, model, criterion, optimizer, 
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train(True)
    batch_time = AverageMeter(name='batch_time')
    data_time = AverageMeter(name='data_time')
    losses = AverageMeter(name='losses')
    accuracies = AverageMeter(name='accuracies')

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = Variable(inputs).to(device)
        targets = Variable(targets).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, inputs.size(0))
        acc = accuracy(outputs.data, targets.data,device=device)
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

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
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc {acc.val:.5f} ({acc.avg:.5f})\t'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      acc=accuracies,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'acc': accuracies.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })
    return losses.avg.item(), accuracies.avg.item()
