from torch.autograd import Variable
import time

from utils import *

def train_epoch(device, epoch, data_loader, model, criterion, optimizer, 
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))

    model.train(True)
    batch_time = AverageMeter(name='batch_time')
    data_time = AverageMeter(name='data_time')
    losses = AverageMeter(name='losses')
    top1 = AverageMeter(name='top1')
    top2 = AverageMeter(name='top2')

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        inputs = inputs.to(device)
        targets = targets.to(device)

        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, inputs.size(0))
        prec1, prec2 = calculate_accuracy(outputs.data, targets.data, topk=(1,2))
        top1.update(prec1, inputs.size(0))
        top2.update(prec2, inputs.size(0))

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
            'prec1': top1.val.item(),
            'prec2': top2.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@2 {top2.val:.5f} ({top2.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top2=top2,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec2': top2.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })
