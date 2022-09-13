import os, sys, numpy as np,  shutil, dicom2nifti
import pandas as pd
import nibabel as nib
from os.path import isfile, isdir, join
from skimage import morphology
from scipy import ndimage
from PIL import Image


# is0001-is0145
# base_path = '/media/john/ischemic stroke/高階影像DICOM/COPY光碟 is0001-154/IS_DWI_CT/3 CT MRI 1-145 DICOM delink/'
# num_listdir = os.listdir(os.path.join(base_path))

# save_path = '/home/john/network/cnn/Fusion_model/dataset/MRI_is001-is350/'
# count = 0
# for i in num_listdir:
#     next_patient = os.path.join(base_path,i,'MR')
#     if isdir(next_patient):
#         patient_lsit = os.listdir(next_patient)
#         if not os.path.exists(os.path.join(save_path, i)):
#             os.makedirs(os.path.join(save_path, i))
#         print('\n------------------', i, 'started!!!\n')
#         for dicom_ in patient_lsit:
#             if isdir(os.path.join(next_patient, dicom_)):
#                 # print(os.path.join(next_patient, dicom_))
#                 dicom2nifti.convert_directory(os.path.join(next_patient, dicom_), os.path.join(save_path, i), compression=True, reorient=True)

# is0155-is0188
# base_path = '/media/john/ischemic stroke/_enrolled dicom 155-350/201907-201912 enrolled is0155-188/'
# num_listdir = os.listdir(os.path.join(base_path))
# # print(num_listdir)
# save_path = '/home/john/network/cnn/Fusion_model/dataset/MRI_is001-is350/'
# for i in num_listdir:
#     next_patient = os.path.join(base_path,i)
#     # print(next_patient)
#     patient_name = i[-6::]
#     # print(patient_name)
#     if isdir(next_patient):
#         patient_lsit = os.listdir(next_patient)
#         if not os.path.exists(os.path.join(save_path, patient_name)):
#             os.makedirs(os.path.join(save_path,patient_name))
#         print('\n------------------', patient_name, 'started!!!\n')
#         dicom2nifti.convert_directory(os.path.join(base_path, i), os.path.join(save_path, patient_name), compression=True, reorient=True)

# is0189-is0232
# base_path = '/media/john/ischemic stroke/_enrolled dicom 155-350/202001-202005 enrolled is0189-232/'
# num_listdir = os.listdir(os.path.join(base_path))
# # print(num_listdir)
# save_path = '/home/john/network/cnn/Fusion_model/dataset/MRI_is001-is350/'
# for i in num_listdir:
#     next_patient = os.path.join(base_path,i)
#     # print(next_patient)
#     patient_name = i[-6::]
#     # print(patient_name)
#     if isdir(next_patient):
#         patient_lsit = os.listdir(next_patient)
#         if not os.path.exists(os.path.join(save_path, patient_name)):
#             os.makedirs(os.path.join(save_path,patient_name))
#         print('\n------------------', patient_name, 'started!!!\n')
#         dicom2nifti.convert_directory(os.path.join(base_path, i), os.path.join(save_path, patient_name), compression=True, reorient=True)

# is0233-is0301
# base_path = '//media/john/ischemic stroke/_enrolled dicom 155-350/202006-202102 enrolled/'
# num_listdir = os.listdir(os.path.join(base_path))
# # print(num_listdir)
# save_path = '/home/john/network/cnn/Fusion_model/dataset/MRI_is001-is350/'
# for i in num_listdir:
#     next_patient = os.path.join(base_path,i)
#     if isdir(next_patient) and ('iagnosis' in next_patient) == False:
#         print(next_patient)
#         patient_name = 'is0' + str(i[-3::])
#         print(patient_name)
#         if not os.path.exists(os.path.join(save_path, patient_name)):
#             os.makedirs(os.path.join(save_path,patient_name))
#         print('\n------------------', patient_name, 'started!!!\n')
#         dicom2nifti.convert_directory(os.path.join(base_path, i), os.path.join(save_path, patient_name), compression=True, reorient=True)

# is0302-is0350
# base_path = '/media/john/ischemic stroke/_enrolled dicom 155-350/202006-202102 enrolled partII/'
# num_listdir = os.listdir(os.path.join(base_path))
# # print(num_listdir)
# save_path = '/home/john/network/cnn/Fusion_model/dataset/MRI_is001-is350/'
# for i in num_listdir:
#     next_patient = os.path.join(base_path,i)
#     if isdir(next_patient) and ('iagnosis' in next_patient) == False:
#         _name = i.replace(' A+P', '')
#         patient_name = 'is0' + str(_name[-3::])
#         print(patient_name)
#         if not os.path.exists(os.path.join(save_path, patient_name)):
#             os.makedirs(os.path.join(save_path,patient_name))
#         print('\n------------------', patient_name, 'started!!!\n')
#         dicom2nifti.convert_directory(os.path.join(base_path, i), os.path.join(save_path, patient_name), compression=True, reorient=True)


source_folder = '/home/john/network/cnn/Fusion_model/dataset/MRI_is001-is350/'
target_folder = '/home/john/network/cnn/Fusion_model/dataset/MRI_is0001-is0350_DWI+T1_2020-01-20/'
# print(filter_csv.columns)
filter_list = ['ssb1000', 't1w_se', 'tracew.nii', 'mprage', 't1_flair_tra.nii', 't1_fl2d_tra.nii', 't1w_ffe_tra.nii']
folder_name = os.listdir('../dataset/MRI_is001-is350/')
for i in folder_name:
    print(i)
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