#패키지
import SimpleITK as sitk
import pydicom as dcm
import numpy as np
from scipy import ndimage
from sklearn.model_selection import train_test_split
import pandas as pd
import preprocessing as ppc

def read_dicom_file(source,filepath):
    """Read and load volume"""
    sitk_t1 = sitk.ReadImage(filepath)
    sitk_t1.SetDirection([1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0])
    image = sitk.GetArrayFromImage(sitk_t1)

    if source=='OASIS-3':
        #Oasis rotate, sort
        image = image.transpose(1,2,0)
        image = ndimage.rotate(image,180,reshape=False)
        image = image[:,:,::-1]
    else: #ADNI
        image = image.transpose(2,0,1)
    return image

def preprocessing(image):
    image = ppc.crop_image(image)
    image = ppc.add_pad(image)
    return image

def process_scan(source, filepath, preprocess= True):
    image = read_dicom_file(source, filepath)
    if preprocess == True:
        image = preprocessing(image)
    return image

def load_dataset(df_dataset,preprocess = True):
    
    img_dataset = np.array([process_scan(source,path,preprocess) for source,path in np.array(df_dataset[['source','path']])])
    
    return img_dataset

def dataset_split(df_dataset,test_size=0.2,shuffle=True,grp=None,seed=1004):
    df_dataset['grp'] = (df_dataset['source'].str.replace('OASIS-3','1').str.replace('ADNI','2').apply(pd.to_numeric)*1000 
                     + df_dataset['sex'].str.replace('F','1').str.replace('M','2').apply(pd.to_numeric)*100
                     + df_dataset['group_maxinc'].str.replace('CN','1').str.replace('MCI','2').str.replace('AD','3').apply(pd.to_numeric)*10
                     #+ (df_dataset['age'] // 10)
                     )
                     
    X = df_dataset.drop(labels='group_maxinc',axis=1)
    Y = df_dataset['group_maxinc']
    grp = df_dataset['grp']

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=test_size,shuffle=True,stratify=grp,random_state=seed)

    return X_train,X_test,y_train,y_test


#Dataset Class 
from torch.utils.data import Dataset
class MRIDataset(Dataset):
    def __init__(self, dataset,labels):
        self.df_train = dataset[['source','path','filename','age']]
        self.df_labels = labels

    def __len__(self):
        return len(self.df_train)

    def __getitem__(self, idx):
        img_path = self.df_train['path'][idx]
        img_source = self.df_train['source'][idx]
        image = process_scan(img_source,img_path)
        label = self.df_labels[idx]
        return image, label