import os, sys, numpy as np,  shutil, dicom2nifti
import pandas as pd
import nibabel as nib
from skimage import morphology
from scipy import ndimage
from PIL import Image

base_path = '/media/john/ischemic stroke/MRA/2021/'
num_listdir = os.listdir(os.path.join(base_path))

save_path = '../dataset/RAW_MRI(MRA new data)/'
count = 0
for i in num_listdir:
        
    for patient in os.listdir(os.path.join(base_path, i)):
        count+=1
        print('\n------------------', patient, 'started!!!\n')
        if not os.path.exists(os.path.join(save_path, patient)):
            os.makedirs(os.path.join(save_path, patient))
        dicom2nifti.convert_directory(os.path.join(base_path, i, patient), os.path.join(save_path, patient), compression=True, reorient=True)
        print('\n',  patient, '------------------finished!!!\n')

filter_csv = pd.read_csv('../csv/20220118_AISlist.csv', index_col=None)
source_folder = '../dataset/RAW_MRI(MRA new data)/'
target_folder = '../dataset/MRI_193964-39010035/'
# print(filter_csv.columns)
filter_list = ['ssb1000', 'ssb 1000', 't1w_se', 'tracew.nii', 'mprage', 'FLAIR_tra']
folder_name = filter_csv['AccessionNumber']
for i in folder_name:
    if not os.path.exists(os.path.join(target_folder, i)):
        os.makedirs(os.path.join(target_folder, i))
    save_dir = os.path.join(target_folder, i)
    try:
        for j in os.listdir(os.path.join(source_folder,i)):
            for ls in filter_list:
                if ls in j and 'nii' in j:
                    shutil.copy(os.path.join(source_folder, i, j), (save_dir+'/'))
                    pass
        print(f"{i} Copy End....")
        pass
    except:
        print(f"{i} Not Found")