from train import train
from validation import validation
import utils
import os
import datetime
import torch.nn as nn
from optimizer import adjust_learning_rate
from torch.optim import lr_scheduler
from model_test import test
import torch

def train_epoch(device,train_dataloader,valid_dataloader,test_dataloader
                ,model,criterion_clf,optimizer
                ,config, epoch,age_onoff=True,num_classes=3):
    
    log_path, store_name = utils.create_storename(config)
    train_logger = utils.Logger(os.path.join(log_path, store_name+'_train.log'),['epoch', 'loss','acc', 'lr'])
    train_batch_logger = utils.Logger(os.path.join(log_path, store_name+'_train_batch.log'), ['epoch', 'batch', 'iter', 'loss','acc', 'lr'])
    valid_logger = utils.Logger(os.path.join(log_path, store_name+'_valid.log'),['epoch', 'loss','acc','ConfusionMatrix','auroc','fpr','tpr','thresholds'])
    
    best_acc = 0
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.7, patience=5)
    for i in range(epoch):
        
        #adjust_learning_rate(optimizer,i,learning_rate,lr_steps=lr_steps)
        
        loss, acc = train(device,i,train_dataloader,model,criterion_clf
                        ,optimizer,train_logger,train_batch_logger,age_onoff=age_onoff)
        
        val_loss,val_acc = validation(device,i,valid_dataloader,model,criterion_clf,valid_logger,age_onoff=age_onoff,num_classes=num_classes)

        
        #성능이 향상이 없을 때 learning rate를 감소시킨다
        scheduler.step(val_acc)
        
        ## model save
        if isinstance(model, nn.DataParallel): ## 다중 GPU를 사용한다면
            state_dict = model.module.state_dict() ## model.module 형태로 module.을 제거하고 저장
        else:
            state_dict = model.state_dict() ## 일반저장

        state = {
                'epoch': i,
                'state_dict': state_dict,
                'optimizer': optimizer.state_dict(),
                'learning_rate': optimizer.param_groups[0]['lr'],
                'valid_loss': val_loss,
                'train_loss': loss,
                'train_acc': acc,
                'valid_loss': val_loss,
                'valid_acc': val_acc
                }

        is_best = val_acc > best_acc
        best_acc = max(val_acc,best_acc)

        #모델 세이브
        utils.save_checkpoint(state, is_best, config)
    
    test_logger = utils.Logger(os.path.join(log_path, store_name+'_test.log'),['best_yn','loss', 'acc','ConfusionMatrix','auroc','fpr','tpr','thresholds'])
    #last model test
    loss, accu, CFM, auroc = test(device,test_dataloader,model,criterion_clf, test_logger, age_onoff = age_onoff,best_yn=False)

    #best model test
    checkpoint = torch.load(os.path.join(log_path, store_name+'_best.pth'))
    if isinstance(model, nn.DataParallel) ==False: ## 다중 GPU를 사용한다면
        model.load_state_dict(checkpoint['state_dict'])
        loss, accu, CFM, auroc = test(device,test_dataloader,model,criterion_clf, test_logger, age_onoff = age_onoff,best_yn=True)    