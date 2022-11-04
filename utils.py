import json
import csv
import matplotlib.pyplot as plt


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
