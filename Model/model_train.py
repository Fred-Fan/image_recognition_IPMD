import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import brewer2mpl
import pandas as pd
import pickle
#import cv2
from Utils import load_pkl_data
from Utils import load_pd_data
from Utils import load_pd_direct

from keras.models import load_model

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report





def build_cnn_model():
    #img_size = 48
    img_size = 96
    img_size_flat = img_size * img_size
    img_shape = (img_size, img_size, 1)
    #num_channels = 1
    num_classes = 8
    # Start construction of the Keras.
    model = Sequential()
    model.add(InputLayer(input_shape=(img_size_flat,)))
    model.add(Reshape(img_shape))

    #model.add(Dropout(0.5, input_shape=(48, 48, 1)))
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))

    model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
                     activation='relu'))
    model.add(Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))


    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    # Last fully-connected / dense layer with softmax-activation
    # for use in classification.
    model.add(Dense(num_classes, activation='softmax'))

    return model


def compile_model(model, optimizer, loss='categorical_crossentropy', metrics=['accuracy']):
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

def save_data():
    return

def train(model, X, Y, epoch, batch_size, callbacks_list):
    model.fit(x = X, y = Y, epochs = epoch, batch_size = batch_size, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=1)
    model.save("checkpoint/mymodel.h5py")
    model_json = model.to_json()
    with open("checkpoint/mymodel.json", "w") as json_file:
        json_file.write(model_json)
    return model

def evaluate(model, X, Y):
    result = model.evaluate(X, Y)
    for name, value in zip(model.metrics_names, result):
        print(name, value)
    return model

def predict(model, X_):
    y_pred = model.predict(x = X_)
    cls_pred = np.argmax(y_pred, axis = 1)
    return y_pred, cls_pred


earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')
callbacks_list = [earlystop]

epochs = 50
batch_size = 256
train_X, train_Y, _, _, train_file = load_pd_data('Data/mtrain96.pd')
test_X, test_Y, _, _, test_file = load_pd_data('Data/mtest96.pd')
#model = build_cnn_model()
model = load_model("Orimodel/mymodel.h5py")
optimizer = Adam(lr = 1e-4)
#model = compile_model(model, optimizer)
model = train(model, train_X, train_Y, epochs, batch_size, callbacks_list)

# evaluate on training set itself
model = evaluate(model, train_X, train_Y)
# predict also on training set itself
y_pred, cls_pred = predict(model, train_X)
print('training accuracy:',cls_pred)
#y_pred, cls_pred = predict(model, test_X)
#print('testing accuracy:',cls_pred)

model.save("Newmodel/mymodel.h5py")
model_json = model.to_json()
with open("Newmodel/mymodel.json", "w") as json_file:
    json_file.write(model_json)
