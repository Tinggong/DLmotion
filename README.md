# DL-based method for motion-robust diffusion kurtosis imaging

## Purpose

This code implements patch-based H-CNN method for motion-robust estimaiton of DKI- and DTI-derived
 diffusion measures described in the paper:

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

Diffusion data preprocessing: FSL at https://fsl.fmrib.ox.ac.uk/fsl/fslwiki

Conventional DKI model fitting: Diffusion Kurtosis estimator (DKE) at https://www.nitrc.org/projects/dke 
or DESIGNER toolbox at https://github.com/NYU-DiffusionMRI/DESIGNER

## Quick-start tutorial

The network model can be easily trained and used with command line inputs after preparing for all necessary files. Documentation for each command can be found using the -h flag. 

### 1. Prepare for your training and testing datasets

a. Data pre-processing: To correct for imaging artefacts and assess motion level for each dataset, preprocessing steps including B0 inhomogeneity, eddy current and motion correction are performed with TOPUP and EDDY in FSL.

Files needed: diffusion.nii, nodif_brain_mask.nii; EDDY log files that are used for quality control: eddy_unwarped_images.eddy_parameters, eddy_unwarped_images.eddy_outlier_map
     
b. Conventional DKI fitting: To generate training labels for the training dataset or reference standards for testing dataset, conventional model-fitting should be performed with full avalible data. Python and bash scripts for DKI fitting with DKE tool is provided. If you use IRLLS fitting with DESIGNER toolbox, the dki_parameters.m needs to be modified to include KFA measure.

Files needed: AD.nii, RD.nii, MD.nii, FA.nii, AK.nii, RK.nii, MK.nii, KFA.nii

c. Data organisation: The dataset should be organised in folders named by the subject ID containing the necessary files for network use.

Mandatory files for training dataset: diffusion.nii, nodif_brain_mask.nii, AD.nii, RD.nii, MD.nii, FA.nii, AK.nii, RK.nii, MK.nii, KFA.nii
Mandatory files for testing dataset: diffusion.nii, nodif_brain_mask.nii and the eddy log files for motion assessment

### 2. Assess motion level for each dataset and generate subject-specific data selection scheme

Please check and make sure your eddy log files contain the same number of measurements as your corrected diffusion data, as it will depend on your acquistion and whether AP and PA combination is used for EDDY.
    
a. Read from eddy log files to generate measures of motion level for each volume of the target data. A txt file 'QAfrom-eddylog.txt' will be generated in the same subject folder, where each row containing the measures of [transform relative to first volume, transform relative to previous volume, rotation relative to first volume, rotation relative to previous volume, percentage of slices with outliers] for each volume.
        
        python3 QAeddy.py --path $SubjDir --eddyname $PrefixName

b. Apply thresholds to each of the motion assessment measures in 'QAfrom-eddylog.txt' to select motion-free volumes for usagage. A file defined by the --schemename will be generated in the same subject folder, containing 1 for the selected image volumes and 0 for all other volumes. 

        python3 SelectScheme.py --path $SubjDir --t0 2 --t1 1.5 --r0 2 --r1 1.5 --out 0.05 --schemename filtered


### 3. Read from nii data and given selection scheme and save data for network use
    
See FormatData.py for details; /datasets folder will be generated in your code folder containing the formatted data. 

a. Formatting full training dataset from traning subjects:
        
        python3 FormatData.py --path $DataDir --subjects S1 S2 --conv3d_train 

b. Apply selection scheme and format test datase: 
        
        python3 FormatData.py --path $DataDir --subjects S3 --scheme filtered --test

### 4. Network training and testing

Check all available options and default values in /utils/model.py

a. To train the 3D H-CNN model with data from S1, the volumes of which are selected from the selection scheme of the test subject S3; "weights" folder will be generated containing the trained model:
        
        python3 Training.py --train_subjects S1 --test_subject S3 --schemename filtered --model conv3d_hcnn --train 
    
b. To apply trained model to the testing data; "nii" folder will be generated containing the estimated measures in nii format:
        
        python3 Testing.py --train_subjects S1 --test_subject S3 --schemename filtered --model conv3d_hcnn 
    
