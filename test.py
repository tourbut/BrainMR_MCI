from torch.autograd import Variable
import time

from metrics import accuracy,roc_auc_score,roc_curve,auc

from utils import *

def test(device, data_loader, model, logger,n_classes=3):
    print('test')

    model.eval()
    accuracies = AverageMeter(name='accuracies')
    
    y_score=[]
    y_test=[]
    for i, (inputs, targets) in enumerate(data_loader):

        with torch.no_grad():
            inputs = Variable(inputs).to(device)
            targets = Variable(targets).to(device)
            
        outputs = model(inputs)
        
        y_test.append(targets.data)
        y_score.append(outputs.data)
        
        acc = accuracy(outputs.data, targets.data,device=device)
        accuracies.update(acc, inputs.size(0))
        print(acc)

    # ROC & AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    score = roc_auc_score(y_test, y_score, multi_class='raise')
    print('acc {acc.avg:.5f}'.format(acc=accuracies))
    print('score : ',score)

    logger.log({
        'acc': accuracies.avg.item(),
        'score': score
    })
    
    return roc_auc, score
