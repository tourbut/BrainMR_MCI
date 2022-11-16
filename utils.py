import json
import csv
import matplotlib.pyplot as plt
import torch
import shutil
import datetime
import os

def load_txt(txt_dir, txt_name):
    List = []
    with open(txt_dir + txt_name, 'r') as f:
        for line in f:
            List.append(line.strip('\n').replace('.nii', '.npy'))
    return List

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    filenames = [a[0] for a in your_list[1:]]
    labels = [0 if a[1]=='CN' else 1 for a in your_list[1:]]
    return filenames, labels

def load_config():
    config = read_json('./config.json')
    return config

def stack_plot(stack,rows=6,cols=6,start_with=10,show_every=5,subtitle='title'):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    plt.suptitle(subtitle)
    for i in range(rows*cols):
        ind = start_with = i*show_every
        ax[int(i / rows),int(i % rows)].set_title('slice %d'%ind)
        ax[int(i / rows),int(i % rows)].imshow(stack[:,:,ind],cmap='gray')
        ax[int(i / rows),int(i % rows)].axis('off')
    plt.show()

def image_plot(img):
    plt.imshow(img,cmap='gray')
    
    
class Logger(object):
    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header
        
    def __del(self):
        self.log_file.close()
        
    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
def create_storename(config):
    log_path = config['result_path']
    log_date = config['save_datetime'] 
    model_name = config['model']['model_name']
    model_depth = config['model']['model_depth']
    store_name = model_name + str(model_depth) +'_' + log_date
    full_path = os.path.join(log_path,store_name)
    os.makedirs(full_path, exist_ok = True)
    
    return full_path,store_name

def save_messgage(config,**kwargs):

    from collections import OrderedDict

    log_path, store_name = create_storename(config)
    path = os.path.join(log_path,store_name+'.json')
    json_data = OrderedDict()

    if kwargs:
        json_data=kwargs

    with open(path, 'w') as outfile:
        json.dump(json_data, outfile)


    
def save_checkpoint(state, is_best, config):

    log_path, store_name = create_storename(config)
    
    torch.save(state, '%s/%s_checkpoint.pth' % (log_path, store_name))
    if is_best:
        shutil.copyfile('%s/%s_checkpoint.pth' % (log_path, store_name),'%s/%s_best.pth' % (log_path, store_name))