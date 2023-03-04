"""
Module definition for network training and testing.
Zhiwei Li, Ting Gong
"""

import os
import argparse
import numpy as np

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, Conv3D, Flatten, Reshape, Conv2DTranspose, UpSampling2D, Concatenate


class MRIModel(object):
    """
    MRI models
    """

    _ndwi = 0
    _single = False
    _model = None
    _type = ''
    _loss = []
    _label = ''
    _kernel1 = 150
    _kernel2 = 150
    _kernel3 = 150

    def __init__(self, ndwi=96, model='conv3d_dki', layer=3, train=True, kernels=None, test_shape=[90, 90, 90]):
        self._ndwi = ndwi
        self._type = model
        self._hist = None
        self._train = train
        self._layer = layer
        self._test_shape = test_shape
        if kernels is not None:
            self._kernel1, self._kernel2, self._kernel3 = kernels

    def _conv2d_model(self, patch_size):
        """
        Conv2D model.
        """
        if self._train:
            inputs = Input(shape=(patch_size, patch_size, self._ndwi))
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, self._ndwi))
        hidden = Conv2D(self._kernel1, 3, strides=1, activation='relu', padding='valid')(inputs)
        for i in np.arange(self._layer - 1):
            hidden = Conv2D(self._kernel1, 1, strides=1, activation='relu', padding='valid')(hidden)
        hidden = Dropout(0.1)(hidden)
        outputs = Conv2D(3, 1, strides=1, activation='sigmoid', padding='valid')(hidden)

        self._model = Model(inputs=inputs, outputs=outputs)

    def _conv3d_model(self, patch_size):
        """
        Conv3D model.
        """
        if self._train:
            inputs = Input(shape=(patch_size, patch_size, patch_size, self._ndwi))
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, dim2, self._ndwi))
        hidden = Conv3D(self._kernel1, 3, activation='relu', padding='valid')(inputs)
        for i in np.arange(self._layer - 1):
            hidden = Conv3D(self._kernel1, 1, activation='relu', padding='valid')(hidden)
        hidden = Dropout(0.1)(hidden)
        outputs = Conv3D(3, 1, activation='sigmoid', padding='valid')(hidden)

        self._model = Model(inputs=inputs, outputs=outputs)

    def _conv2d_staged_model(self, patch_size):
        """
        2D H-CNN model.
        """

        if self._train:
            inputs = Input(shape=(patch_size, patch_size, self._ndwi))
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, self._ndwi))
        hidden = Conv2D(self._kernel1, 3, strides=1, activation='relu', padding='valid')(inputs)
        hidden = Conv2D(self._kernel2, 1, strides=1, activation='relu', padding='valid')(hidden)

        middle = Dropout(0.1)(hidden)
        y = Conv2D(3, 1, strides=1, activation='relu', padding='valid', name='y')(middle)
        hidden = Conv2D(self._kernel3, 1, strides=1, activation='relu', padding='valid')(hidden)

        hidden = Dropout(0.1)(hidden)
        outputs = Conv2D(5, 1, strides=1, activation='relu', padding='valid', name='output')(hidden)

        self._model = Model(inputs=inputs, outputs=[y, outputs])
  
    def _conv3d_staged_model(self, patch_size, convs=None):
        """
        3D H-CNN model.
        """
        if self._train:
            inputs = Input(shape=(patch_size, patch_size, patch_size, self._ndwi))
        else:
            (dim0, dim1, dim2) = (self._test_shape[0], self._test_shape[1], self._test_shape[2])
            inputs = Input(shape=(dim0, dim1, dim2, self._ndwi))
        hidden = Conv3D(self._kernel1, 3, strides=1, activation='relu', padding='valid')(inputs)
        hidden = Conv3D(self._kernel2, 1, strides=1, activation='relu', padding='valid')(hidden)

        if self._layer == 4:
            hidden = Conv3D(self._kernel2, 1, strides=1, activation='relu', padding='valid')(hidden)

        middle = Dropout(0.1)(hidden)
        y = Conv3D(3, 1, strides=1, activation='relu', padding='valid', name='y')(middle)
        hidden = Conv3D(self._kernel3, 1, strides=1, activation='relu', padding='valid')(hidden)

        hidden = Dropout(0.1)(hidden)
        outputs = Conv3D(5, 1, strides=1, activation='relu', padding='valid', name='output')(hidden)

        self._model = Model(inputs=inputs, outputs=[y, outputs])

    __model = {
        'conv2d_noddi': _conv2d_model,
        'conv3d_noddi' : _conv3d_model,
        'conv2d_dki': _conv2d_staged_model,
        'conv3d_dki' : _conv3d_staged_model,
    }

    def model(self, optimizer, loss, patch_size):
        """
        Generate model.
        """
        self.__model[self._type](self, patch_size)
        self._model.summary()
        self._model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    def _sequence_train(self, data, label, nbatch, epochs, callbacks, shuffle, validation_data):

        validation_split = 0.0
        if validation_data is None:
            validation_split = 0.2

        self._hist = self._model.fit(data, label,
                                     batch_size=nbatch,
                                     epochs=epochs,
                                     shuffle=shuffle,
                                     validation_data=validation_data,
                                     validation_split=validation_split,
                                     callbacks=callbacks)
        self._loss.append(len(self._hist.history['loss']))
        self._loss.append(self._hist.history['loss'][-1])
        self._loss.append(None)
        self._loss.append(self._hist.history['accuracy'][-1])
        self._loss.append(None)

    def _staged_train(self, data, label, nbatch, epochs, callbacks, shuffle, validation_data):

        validation_split = 0.0
        if validation_data is not None:
            vdata, vlabel = validation_data
            validation_data = (vdata, [vlabel[..., :3], vlabel[..., 3:]])
        else:
            validation_split = 0.2

        self._hist = self._model.fit(data, [label[..., :3], label[..., 3:]],
                                     batch_size=nbatch,
                                     epochs=epochs,
                                     shuffle=shuffle,
                                     validation_data=validation_data,
                                     validation_split=validation_split,
                                     callbacks=callbacks)
        self._loss.append(len(self._hist.history['loss']))
        self._loss.append(self._hist.history['y_accuracy'][-1])
        self._loss.append(self._hist.history['output_accuracy'][-1])
        self._loss.append(self._hist.history['y_loss'][-1])
        self._loss.append(self._hist.history['output_loss'][-1])

    __train = {
        'conv2d_noddi': _sequence_train,
        'conv3d_noddi' : _sequence_train,
        'conv2d_dki': _staged_train,
        'conv3d_dki' : _staged_train,
    }

    def train(self, data, label, nbatch, epochs, callbacks, weightname,
              shuffle=True, validation_data=None):
        """
        Training on training datasets.
        """
        #print "Training start ..."
        self.__train[self._type](self, data, label, nbatch, epochs,
                                 callbacks, shuffle, validation_data)

        try:
            self._model.save_weights('weights/' + weightname + '.weights')
        except IOError:
            os.system('mkdir weights')
            self._model.save_weights('weights/' + weightname + '.weights')

        return self._loss

    def load_weight(self, weightname):
        """
        Load pre-trained weights.
        """
        self._model.load_weights('weights/' + weightname + '.weights')

    def predict(self, data):
        """
        Predict on test data.
        """
        pred = self._model.predict(data)
        if self._type[-3:] == 'dki':
            pred = np.concatenate((pred[0], pred[1]), axis=-1)

        return pred

def parser():
    """
    Create a parser.
    """
    parser = argparse.ArgumentParser()
    
    # Specify train & test sets
    parser.add_argument("--train_subjects", help="Training subjects IDs", nargs='*')
    parser.add_argument("--test_subject", help="Testing subject ID", nargs='*')
    parser.add_argument("--schemename", metavar='name', help="The scheme for sampling", default='first')
 
   # Training parameters
    parser.add_argument("--train", help="Train the network", action="store_true")
    parser.add_argument("--model", help="Train model",
                        choices=['conv3d_dki','conv2d_dki','conv3d_noddi', 'conv2d_noddi'], default='conv3d_dki')
    parser.add_argument("--epoch", metavar='ep', help="Number of epoches", type=int, default=100)
    parser.add_argument("--lr", metavar='lr', help="Learning rates", type=float, default=0.001)
    
    parser.add_argument("--layer", metavar='l', help="Number of layers for sequence model", type=int, default=3)
    parser.add_argument("--kernels", help="The number of kernels for each layer", nargs='*',
                        type=int, default=None)        
    parser.add_argument("--loss", help="Set different loss functions", type=int, default=0)
    parser.add_argument("--test_shape", nargs='*', type=int, default=None)
    parser.add_argument("--batch", metavar='bn', help="Batch size", type=int, default=256)
    parser.add_argument("--patch_size", metavar='ksize', help="Size of the kernels", type=int, default=3)
    parser.add_argument("--base", metavar='base', help="choice of training data", type=int, default=1)    

    return parser
