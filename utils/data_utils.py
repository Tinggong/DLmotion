"""
Functions for generating and saving data new datasets, and fetching training datasets
Ting Gong

"""

import os
import numpy as np
from scipy.io import loadmat, savemat
from utils.nii_utils import load_nii_image, save_nii_image, mask_nii_data

def gen_dMRI_fc1d_train_datasets(path, subject, ltype, whiten=False):
    """
    Generate fc1d training Datasets.
    """
    os.system("mkdir -p datasets/data datasets/label datasets/mask")
    os.system('cp ' +  path + '/' + subject + '/nodif_brain_mask.nii datasets/mask/mask_' + subject + '.nii')      
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
        
    # load diffusion data
    data = load_nii_image(path + '/' + subject + '/diffusion.nii', mask)

    # Whiten the data.
    if whiten:
        data = data / data.mean() - 1.0
    print(data.shape)

    # load labels
    label = np.zeros((data.shape[0] , len(ltype)))

    for i in range(len(ltype)):
        filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
        temp = load_nii_image(filename,mask) 
        label[:, i] = temp.reshape(temp.shape[0])  
     
    print(label.shape)

    # save datasets
    savemat('datasets/data/' + subject + '-' + 'all' + '-' + 'fullsample' + '-' + '1d.mat', {'data':data})
    savemat('datasets/label/' + subject + '-' + 'all' + '-' + 'fullsample' + '-' + '1d.mat', {'label':label})

def gen_2d_patches(data, mask, size, stride):
    """
    generate 2d patches
    """
    patches = []
    for layer in range(mask.shape[2]):
        for x in np.arange(0, mask.shape[0], stride):
            for y in np.arange(0, mask.shape[1], stride):
                xend, yend = np.array([x, y]) + size
                lxend, lyend = np.array([x, y]) + stride
                if mask[x:lxend, y:lyend, layer].sum() > 0:
                    patches.append(data[x:xend, y:yend, layer, :])

    return np.array(patches)

def gen_3d_patches(data, mask, size, stride):
    """
    generate 3d patches
    """
    #print data.shape, mask.shape
    patches = []
    for layer in np.arange(0, mask.shape[2], stride):
        for x in np.arange(0, mask.shape[0], stride):
            for y in np.arange(0, mask.shape[1], stride):
                xend, yend, layerend = np.array([x, y, layer]) + size
                lxend, lyend, llayerend = np.array([x, y, layer]) + stride
                if mask[x:lxend, y:lyend, layer:llayerend].sum() > 0:
                    patches.append(data[x:xend, y:yend, layer: layerend, :])
    #print np.array(patches).shape
    return np.array(patches)

def gen_dMRI_conv2d_train_datasets(path, subject, ltype, patch_size, label_size, base=1, test=False):
    """
    Generate Conv2D Dataset.
    """
    offset = base - (patch_size - label_size) / 2

    os.system("mkdir -p datasets/data datasets/label datasets/mask")
    os.system('cp ' +  path + '/' + subject + '/nodif_brain_mask.nii datasets/mask/mask_' + subject + '.nii')      
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')

    # load diffusion data
    data = load_nii_image(path + '/' + subject + '/diffusion.nii')
    data = data[:, :, base:-base, :]
   
    #if offset:
    #    data = data[offset:-offset, offset:-offset, :, :]
    
    # load labels
    label = np.zeros(mask.shape + (len(ltype),))
    for i in range(len(ltype)):
        filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
        label[:, :, :, i] = load_nii_image(filename) 

    label = label[base:-base, base:-base, base:-base, :]
    mask = mask[base:-base, base:-base, base:-base]
 
    patches = gen_2d_patches(data, mask, patch_size, label_size)
    label = gen_2d_patches(label, mask, label_size, label_size)
    print(patches.shape)
    print(label.shape)

    savemat('datasets/data/' + subject + '-' + 'all' + '-' + 'fullsample' + '-' + '2d.mat', {'data':patches})
    savemat('datasets/label/' + subject + '-' + 'all' + '-' + 'fullsample' + '-' + '2d.mat', {'label':label})

def gen_dMRI_conv3d_train_datasets(path, subject, ltype, patch_size, label_size, base=1, test=False):
    """
    Generate Conv3D Dataset.
    """
    offset = base - (patch_size - label_size) / 2

    os.system("mkdir -p datasets/data datasets/label datasets/mask")
    os.system('cp ' +  path + '/' + subject + '/nodif_brain_mask.nii datasets/mask/mask_' + subject + '.nii')      
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')

    # load diffusion data
    data = load_nii_image(path + '/' + subject + '/diffusion.nii')
    #if offset:
    #    data = data[offset:-offset, offset:-offset, :, :]
    
    # load labels
    label = np.zeros(mask.shape + (len(ltype),))
    for i in range(len(ltype)):
        filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
        label[:, :, :, i] = load_nii_image(filename) 
    label = label[base:-base, base:-base, base:-base, :]
    mask = mask[base:-base, base:-base, base:-base]
 
    patches = gen_3d_patches(data, mask, patch_size, label_size)
    label = gen_3d_patches(label, mask, label_size, label_size)
    print(patches.shape)
    print(label.shape)

    savemat('datasets/data/' + subject + '-' + 'all' + '-' + 'fullsample' + '-' + '3d.mat', {'data':patches})
    savemat('datasets/label/' + subject + '-' + 'all' + '-' + 'fullsample' + '-' + '3d.mat', {'label':label})

def gen_dMRI_test_datasets(path, subject, ndwi, scheme, combine, ltype, fdata=True, flabel=False, whiten=True):
    """
    Generate testing Datasets.
    """
    os.system("mkdir -p datasets/data datasets/label datasets/mask datasets/scheme")
    os.system('cp ' +  path + '/' + subject + '/nodif_brain_mask.nii datasets/mask/mask_' + subject + '.nii') 
    os.system('cp ' +  path + '/' + subject + '/' + scheme + ' datasets/scheme/' + scheme + '_' + subject)   
    mask = load_nii_image('datasets/mask/mask_' + subject + '.nii')
            
    if fdata:
        data = load_nii_image(path + '/' + subject + '/diffusion.nii')
        
        # Select the inputs.
        if combine is not None:
            data = data[..., combine == 1]
        else:
            data = data[..., :ndwi]

        # Whiten the data.
        if whiten:
            data = data / data[mask > 0].mean() - 1.0
        
        print(data.shape)
        savemat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '.mat', {'data':data})

    if flabel:
        label = np.zeros(mask.shape + (len(ltype),))

        for i in range(len(ltype)):
            filename = path + '/' + subject + '/' + subject + '_' + ltype[i] + '.nii'
            label[:, :, :, i] = load_nii_image(filename)
            savemat('datasets/label/' + subject+ '-' + str(ndwi) + '-' + scheme + '.mat', {'label':label})

def fetch_train_data_MultiSubject(subjects, model, ndwi, scheme, combine, whiten=True):
    """
    #Fetch train data.
    """
    data_s = None
    labels = None

    if model[:4] == 'fc1d':
        dim='1d.mat'
    if model[:6] == 'conv2d':
        dim='2d.mat'
    if model[:6] == 'conv3d':
        dim='3d.mat'


    for subject in subjects:
        #label = loadmat('datasets/label/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + dim)['label']
        #data = loadmat('datasets/data/' + subject + '-' + str(ndwi) + '-' + scheme + '-' + dim)['data']
        label = loadmat('datasets/label/' + subject + '-' + 'all' + '-' + 'fullsample' + '-' + dim)['label']
        data = loadmat('datasets/data/' + subject + '-' + 'all' + '-' + 'fullsample' + '-' + dim)['data']

        # Select the inputs.
        if combine is not None:
            data = data[..., combine == 1]
        else:
            data = data[..., :ndwi]

        # Exclude samples with nan values in label
        for i in range(label.shape[0]):
            if np.isnan(label[i]).any():
                label[i] = 0
                data[i] = 0
    
        # Whiten the data.
        if whiten:
            data = data / data.mean() - 1.0

        if data_s is None:
            data_s = data
            labels = label
        else:
            data_s = np.concatenate((data_s, data), axis=0)
            labels = np.concatenate((labels, label), axis=0)

    data = np.array(data_s)
    label = np.array(labels)

    return data, label

def shuffle_data(data, label):
    """
    Shuffle data.
    """
    size = data.shape[-1]
    datatmp = np.concatenate((data, label), axis=-1)
    np.random.shuffle(datatmp)
    return datatmp[..., :size], datatmp[..., size:]

def repack_pred_label(pred, mask, model, ntype):
    """
    Get.
    """
    if model[7:13] == 'single':
        label = np.zeros(mask.shape + (1,))
    else:
        label = np.zeros(mask.shape + (ntype,))
    
    if model[:6] == 'conv2d':
        label[1:-1, 1:-1, :, :] = pred.transpose(1, 2, 0, 3)
    elif model[:6] == 'conv3d':
        label[1:-1, 1:-1, 1:-1, :] = pred
    else:
        label = pred.reshape(label.shape)
    
    return label