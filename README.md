# DL-based method for motion-robust diffusion parameter estimation

## Purpose

This code implements patch-based CNN method for reducing residual motion effects in diffusion parameter estimation described in the paper:

[1] Gong T, Tong Q, Li Z, He H, Zhang H, Zhong J. Deep learning-based method for reducing residual
    motion effects in diffusion parameter estimation. Magn Reson Med. 2020;00:1–16.

## Dependencies

The current version is implemented in Python 3.7 using Keras with Tensorflow backend.

### 1. Packages needed for the H-CNN model:
    # Install tensorflow (which now includes keras) these two librarys are used for deep learning in python
    pip3 install tensorflow==2.3.1

    # Install scipy and numpy these libraries are used for performing mathmatical calculations on datasets 
    pip3 install scipy
    pip3 install numpy==1.17.0

    # Install nipy, this library is used for loading NIfTI images (.nii/.nii.gz). This is how MRI images are normally saved
    pip3 install nipy==0.4.2


### 2. Other tools useful for data preparation（please cite corresponding papers suggested if you use them):

a. Tool for diffusion data preprocessing: FSL at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

b. Tool for conventional model fitting. 

        1. DKI model: This tutoral uses DKI model as a demonstration and following tool can be found: 
        Diffusion Kurtosis estimator (DKE) at https://www.nitrc.org/projects/dke 
        or DESIGNER toolbox at https://github.com/NYU-DiffusionMRI/DESIGNER

        2. NODDI model: matlab toolbox can be found at https://www.nitrc.org/projects/noddi_toolbox 

## Quick-start tutorial

The network model can be easily trained and used with command line inputs after preparing for all necessary files. Documentation for each command can be found using the -h flag. 

### 1. Prepare your datasets

a. Data pre-processing: To correct for imaging artefacts and assess motion level for each dataset, preprocessing steps including B0 inhomogeneity, eddy current and motion correction are performed with TOPUP and EDDY in FSL.

Generated files in subject folder: 
diffusion.nii        # the 4D diffusion dataset
nodif_brain_mask.nii # the binary brain mask
bval                 # the b-values associated with each image volume
bvec                 # the directions of diffusion gradients
eddy_unwarped_images.eddy_parameters     # EDDY log files that are used for quality control; translations and rotations 
eddy_unwarped_images.eddy_outlier_map    # EDDY log files that are used for quality control; outliers 
     
b. Conventional model fitting for training dataset: To generate training labels for the training dataset, conventional model-fitting should be performed with full avalible data. 

Generated training labels in training subject folder: 
For DKI model: SubjID_AD.nii, SubjID_RD.nii, SubjID_MD.nii, SubjID_FA.nii, SubjID_AK.nii, SubjID_RK.nii, SubjID_MK.nii, SubjID_KFA.nii
For NODDI model: SubjID_ficvf.nii, SubjID_fiso.nii, SubjID_odi.nii

c. Data organisation. The dataset should be organised in folders named by the subject ID containing the necessary files for network use.

Mandatory files for training dataset: diffusion.nii, nodif_brain_mask.nii, and training labels.
Mandatory files for other datasets: diffusion.nii, nodif_brain_mask.nii and the eddy log files for motion assessment

### 2. Assess motion level for each dataset and generate subject-specific data selection scheme

Please check and make sure your eddy log files contain the same number of measurements as your corrected diffusion data, as it will depend on your acquistion and whether AP and PA combination is used for EDDY.
    
a. Read from eddy log files to generate measures of motion level for each volume of the target data:
        
        python3 QAeddy.py --path $SubjDir --eddyname eddy_unwarped_images

        A txt file 'QAfrom-eddylog.txt' will be generated in the same subject folder, where each row containing the following measures of each volume (t0, t1, r0, r1, outlier): 
        (transform relative to first volume, transform relative to previous volume, rotation relative to first volume, rotation relative to previous volume, percentage of slices with outliers)

b. Apply thresholds to each of the motion assessment measures in 'QAfrom-eddylog.txt' to select motion-free volumes for usagage. A file defined by the --schemename will be generated in the same subject folder, containing 1 for the selected image volumes and 0 for all other volumes. 

        python3 SelectScheme.py --path $SubjDir --t0 2 --t1 1.5 --r0 2 --r1 1.5 --outlier 0.05 --schemename filtered

        This is an example of using thresholds of [t0, t1, r0, r1, outlier]=[2 mm, 1.5 mm, 2 degree, 1.5 degree, 5%] as defined above.

### 3. Read from nii data and given selection scheme and save data for network use
    
See FormatData.py for details; /datasets folder will be generated in your code folder containing the formatted data. 

a. Formatting full training dataset from traning subjects (dki model as an example):
        
        python3 FormatData.py --path $DataDir --subjects S1 S2 --diffmodel dki --conv3d_train 

b. Apply selection scheme to each study dataset: 
        
        python3 FormatData.py --path $DataDir --subjects S3 --scheme filtered --test

### 4. Train the subject-specific network model and apply the model to the study subject

Check all available options and default values in /utils/model.py

a. This example trains the 3D H-CNN model for DKI with data from S1, the volumes of which are selected from the selection scheme of the study subject S3; the trained model is then applied to data of S3.
"weights" folder will be generated containing the trained model;
"nii" folder will be generated containing the estimated measures in nii format.
        
        python3 dMRInet.py --train_subjects S1 --test_subject S3 --schemename filtered --model conv3d_dki --train 
    
    
