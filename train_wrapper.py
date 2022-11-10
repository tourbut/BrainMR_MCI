from train import train
from validation import validation
import utils
import os
import datetime

def train_epoch(device,train_dataloader,valid_dataloader,model,criterion_clf,optimizer,config, epoch = 100):
    
    log_path = config['log_path']
    log_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    store_name = config['store_name']
    store_name = store_name+'_'+log_date

    train_logger = utils.Logger(os.path.join(log_path, store_name+'_train.log'),['epoch', 'loss','acc', 'lr'])
    train_batch_logger = utils.Logger(os.path.join(log_path, store_name+'_train_batch.log'), ['epoch', 'batch', 'iter', 'loss','acc', 'lr'])
    valid_logger = utils.Logger(os.path.join(log_path, store_name+'_valid.log'),['epoch', 'loss','acc'])
    
    best_acc = 0
    
    for i in range(epoch):
        model, loss, acc = train(device,i,train_dataloader,model,criterion_clf,optimizer,train_logger,train_batch_logger)
        
        state = {
                'epoch': i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'acc': acc
                }
        
        val_loss,val_acc = validation(device,i,valid_dataloader,model,criterion_clf,valid_logger)
        
        is_best = val_acc > best_acc
        best_acc = max(val_acc,best_acc)

        #모델 세이브
        utils.save_checkpoint(state, best_acc, config)
        