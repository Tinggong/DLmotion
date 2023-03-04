"""
Read from eddy log files to generate measures of motion level for each volume of the target data.

A txt file 'QAfrom-eddylog.txt' will be generated in the same subject path, where each row containing the measures of 
[transform relative to first volume, transform relative to previous volume, rotation relative to first volume, rotation relative to previous volume, percentage of slices with outliers] 
for each volume.

Usage: python QAeddy.py --path $SubjDir --eddyname $PrefixName

Author: Zhiwei Li, Ting Gong
"""

import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="The path of folder containing eddy log files")
parser.add_argument("--eddyname", help="Prefix of eddy log files")

args = parser.parse_args()

path = args.path
eddyname = args.eddyname

filename = path + '/' + eddyname + '.eddy_parameters'
eddyparas = np.loadtxt(filename) 

filename = path + '/' + eddyname + '.eddy_outlier_map'
outlierparas = np.loadtxt(filename, skiprows = 1) 

affine2b0 = eddyparas[:,0:6]
affine2form = affine2b0.copy()
affine2form[1:,:] = affine2form[1:,:] - affine2form[0:-1,:]

trans2b0 = np.sqrt(np.sum(np.square(affine2b0[:,0:3]),axis=1))
trans2form = np.sqrt(np.sum(np.square(affine2form[:,0:3]),axis=1))

rot2b0 = np.sum(np.abs(affine2b0[:,3:6])*180/np.pi,axis=1)
rot2form = np.sum(np.abs(affine2form[:,3:6]*180/np.pi),axis=1)

sliceperct = np.mean(outlierparas,axis=1)

np.savetxt(path + '/' + 'QAfrom-eddylog.txt', np.stack((trans2b0,trans2form,rot2b0,rot2form,sliceperct),axis=1))






