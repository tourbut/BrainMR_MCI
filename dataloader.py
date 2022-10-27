#패키지
import SimpleITK as sitk
import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.model_selection import train_test_split
import pandas as pd

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

    return image

def process_scan(source, filepath):
    image = read_dicom_file(source, filepath)
    image = preprocessing(image)
    return image

def sample_stack(stack,rows=6,cols=6,start_with=10,show_every=5,subtitle='title'):
    fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    plt.suptitle(subtitle)
    for i in range(rows*cols):
        ind = start_with = i*show_every
        ax[int(i / rows),int(i % rows)].set_title('slice %d'%ind)
        ax[int(i / rows),int(i % rows)].imshow(stack[:,:,ind],cmap='gray')
        ax[int(i / rows),int(i % rows)].axis('off')
    plt.show()

def load_dataset(df_dataset):
    img_dataset = np.array([process_scan(source,path) for source,path in np.array(df_dataset[['source','path']])])
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

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,shuffle=True,stratify=grp,random_state=seed)

    return X_train,X_test,y_train,y_test

