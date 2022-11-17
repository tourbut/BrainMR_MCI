from models import resnet
import torch
import torch.nn as nn
import torch.optim as optim

def generate_model(model_name = 'resnet',model_depth = 10,n_classes=3,resnet_shortcut='B',add_last_fc_num = 0):
    '''
        model_name = resnet,
        model_depth = 10,
        n_classes=3,
        resnet_shortcut='B',
        sample_size=112,
        sample_duration=16
    '''
    if model_depth == 10:
        model = resnet.resnet10(
            num_classes=n_classes,
            shortcut_type=resnet_shortcut,
            add_last_fc_num = add_last_fc_num)
    elif model_depth == 18:
        model = resnet.resnet18(
            num_classes=n_classes,
            shortcut_type=resnet_shortcut,
            add_last_fc_num = add_last_fc_num)
    elif model_depth == 34:
        model = resnet.resnet34(
            num_classes=n_classes,
            shortcut_type=resnet_shortcut,
            add_last_fc_num = add_last_fc_num)
    elif model_depth == 50:
        model = resnet.resnet50(
            num_classes=n_classes,
            shortcut_type=resnet_shortcut,
            add_last_fc_num = add_last_fc_num)
    elif model_depth == 101:
        model = resnet.resnet101(
            num_classes=n_classes,
            shortcut_type=resnet_shortcut,
            add_last_fc_num = add_last_fc_num)
    elif model_depth == 152:
        model = resnet.resnet152(
            num_classes=n_classes,
            shortcut_type=resnet_shortcut,
            add_last_fc_num = add_last_fc_num)
    elif model_depth == 200:
        model = resnet.resnet200(
            num_classes=n_classes,
            shortcut_type=resnet_shortcut)
    
    model.name = model_name+str(model_depth)

    return model, model.parameters()