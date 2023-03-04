"""
Main script for network training
Definition of the command-line arguments are in model.py and can be displayed by `python dMRInet.py -h`

Author: Zhiwei Li, Ting Gong 
"""

import numpy as np
import os
import time

from scipy.io import savemat, loadmat

from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, \
                                                            EarlyStopping

from utils import save_nii_image, calc_RMSE, loss_func, repack_pred_label, \
                  MRIModel, parser, load_nii_image, unmask_nii_data, loss_funcs, fetch_train_data_MultiSubject


# Get parameter from command-line input
args = parser().parse_args()

train_subjects = args.train_subjects
test_subject = args.test_subject[0]

# determin the input volumes with a scheme file from the testing subject
combine = None
scheme = args.schemename
if scheme is not None:
    combine = np.loadtxt('datasets/scheme/' + scheme + '_' + test_subject)
    combine = combine.astype(int)
    nDWI = combine.sum()

mtype = args.model
train = args.train

lr = args.lr
epochs = args.epoch
kernels = args.kernels
layer = args.layer

loss = args.loss
batch_size = args.batch
patch_size = args.patch_size
label_size = patch_size - 2
base = args.base

# Parameter name definition
savename = test_subject + str(nDWI)+ '-'  + scheme + '-' + args.model

# Constants
if mtype[-3:] == 'dki':
    types = ['MD' , 'AD' , 'RD' , 'FA' , 'MK' , 'AK', 'RK', 'KFA']
if mtype[-5:] == 'noddi':
    types = ['ficvf' , 'fiso' , 'odi']
ntypes = len(types)
decay = 0.1

shuffle = False
y_accuracy = None
output_accuracy = None
y_loss = None
output_loss = None
nepoch = None

# Define the adam optimizer
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

# Train on the training data.
if train:
    # Define the model.
    model = MRIModel(nDWI, model=mtype, layer=layer, train=train, kernels=kernels)

    model.model(adam, loss_funcs[loss], patch_size)

    data, label = fetch_train_data_MultiSubject(train_subjects, mtype, nDWI, scheme, combine, whiten=True)

    reduce_lr = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=10, epsilon=0.0001)
    tensorboard = TensorBoard(histogram_freq=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=30, min_delta=0.0000005)

    [nepoch, output_loss, y_loss, output_accuracy, y_accuracy] = model.train(data, label, batch_size, epochs,
                                                                   [reduce_lr, tensorboard, early_stop],
                                                                   savename, shuffle=not shuffle,
                                                                   validation_data=None)

# Load testing data
mask = load_nii_image('datasets/mask/mask_' + test_subject + '.nii')
tdata = loadmat('datasets/data/' + test_subject + '-' + str(nDWI) + '-' + scheme + '.mat')['data']

# Reshape the data to suit the model.
if mtype[:6] == 'conv3d':
  tdata = np.expand_dims(tdata, axis=0)
elif mtype[:6] == 'conv2d':
  tdata = tdata.transpose((2, 0, 1, 3))

test_shape = args.test_shape
if test_shape is None:
  test_shape = tdata.shape[1:4]

# Define the model
model = MRIModel(nDWI, model=mtype, layer=layer, train=False, kernels=kernels, test_shape=test_shape)
model.model(adam, loss_func, patch_size)
model.load_weight(savename)

weights = model._model.layers[1].get_weights()

# Predict on the test data.
time1 = time.time()
pred = model.predict(tdata)
print(pred.shape)
time2 = time.time()

time3 = time.time()
pred = repack_pred_label(pred, mask, mtype, ntypes)
time4 = time.time()

#print "predict done", time2 - time1, time4 - time3

# Save estimated measures to /nii folder as nii image
os.system("mkdir -p nii")

for i in range(ntypes):
    data = pred[..., i]
    filename = 'nii/' + test_subject + '-' + types[i] + '-' + savename + '.nii'

    data[mask == 0] = 0
    save_nii_image(filename, data, 'datasets/mask/mask_' + test_subject + '.nii', None)
