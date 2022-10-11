"""
Read from nii data and given under-samping scheme and save .mat for network use

Usage:
1. Format the training dataset containing all the diffusion image volumes and training lables, which allows you
    to select any sub-sampling schemes for training a model. This step needs to be done only once. Depending on 
    the size of inputs (2d patches or 3d patches) and diffusion models (DKI or NODDI) you use, choose different method: 
    
    python FormatData.py --path $DataDir --subjects TrainSubjID1  TrainSubjID2 --diffmodel dki --conv3d_train 

2. Format the testing dataset with a selection scheme after motion assessment. 
    This step needs to be performed for each test subject and when you have a new selection for a subject: 
    
    python FormatData.py --path $DataDir --subjects TestSubjID1 --schemename scheme --test 


Author: Ting Gong
"""
import argparse
import numpy as np

from utils import gen_dMRI_test_datasets, gen_dMRI_fc1d_train_datasets, gen_dMRI_conv2d_train_datasets, gen_dMRI_conv3d_train_datasets


parser = argparse.ArgumentParser()
parser.add_argument("--path", help="The path of data folder")
parser.add_argument("--subjects", help="subjects ID", nargs='*')
parser.add_argument("--schemename", help="The sampling scheme used")
parser.add_argument("--diffmodel", help="Diffusion model used for training")
parser.add_argument("--conv2d_train", help="generate 2d patches for training", action="store_true")
parser.add_argument("--conv3d_train", help="generate 3d patches for training", action="store_true")
parser.add_argument("--test", help="generate base data for testing", action="store_true")

args = parser.parse_args()

path = args.path
subjects = args.subjects
#fc1d_train = args.fc1d_train
test = args.test
diffmodel = args.diffmodel

conv2d_train = args.conv2d_train
conv3d_train = args.conv3d_train
patch_size = 3
label_size = 1
scheme = args.schemename
if diffmodel == 'dki':
    ltype = ['MD' , 'AD' , 'RD' , 'FA' , 'MK' , 'AK', 'RK', 'KFA']
if diffmodel == 'noddi':
    ltype = ['ficvf' , 'fiso' , 'odi']

if test:
    for subject in subjects:

        # determin the input volumes with a scheme file for testing dataset
        combine = None
        if scheme is not None:
            combine = np.loadtxt(path + '/' + subject +'/' + scheme)
            combine = combine.astype(int)
            nDWI = combine.sum()
        gen_dMRI_test_datasets(path, subject, nDWI, scheme, combine, ltype=None, fdata=True, flabel=False, whiten=True)

#if fc1d_train:
#    for subject in subjects:
#        gen_dMRI_fc1d_train_datasets(path, subject, ltype, whiten=True)

if conv2d_train:
    for subject in subjects:
        gen_dMRI_conv2d_train_datasets(path, subject, ltype, patch_size, label_size, base=1, test=False)

if conv3d_train:
    for subject in subjects:
        gen_dMRI_conv3d_train_datasets(path, subject, ltype, patch_size, label_size, base=1, test=False)
