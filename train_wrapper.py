from train import train
from validation import validation
import utils
import os
import datetime
import torch.nn as nn
from optimizer import adjust_learning_rate

def train_epoch(device,train_dataloader,valid_dataloader,model,criterion_clf,optimizer,config, epoch,learning_rate,lr_steps):
    
    log_path = config['log_path']
    log_date = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    model_name = config['model']['model_name']
    model_depth = config['model']['model_depth']
    store_name = model_name + str(model_depth)
    store_name = store_name+'_'+log_date

    train_logger = utils.Logger(os.path.join(log_path, store_name+'_train.log'),['epoch', 'loss','acc', 'lr'])
    train_batch_logger = utils.Logger(os.path.join(log_path, store_name+'_train_batch.log'), ['epoch', 'batch', 'iter', 'loss','acc', 'lr'])
    valid_logger = utils.Logger(os.path.join(log_path, store_name+'_valid.log'),['epoch', 'loss','acc'])
    
    best_acc = 0
    
    for i in range(epoch):
        
        adjust_learning_rate(optimizer,i,learning_rate,lr_steps=lr_steps)
        
        model, loss, acc = train(device,i,train_dataloader,model,criterion_clf,optimizer,train_logger,train_batch_logger)
        
        ## model save
        if isinstance(model, nn.DataParallel): ## 다중 GPU를 사용한다면
            state_dict = model.module.state_dict() ## model.module 형태로 module.을 제거하고 저장
        else:
            state_dict = model.state_dict() ## 일반저장

        state = {
                'epoch': i,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'acc': acc
                }
        
        val_loss,val_acc = validation(device,i,valid_dataloader,model,criterion_clf,valid_logger)
        
        is_best = val_acc > best_acc
        best_acc = max(val_acc,best_acc)

        #모델 세이브
        utils.save_checkpoint(state, is_best, config)
        