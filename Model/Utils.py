import numpy as np
import math
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import time
import json
from keras.models import Sequential


# load data in imagelist.pkl format
def load_pkl_data(file_path):
    with open(file_path, 'rb') as fin:
        kaggle_pkl = pickle.load(fin)
        kaggle_img = np.asarray([item[1] for item in kaggle_pkl])
        kaggle_label = np.asarray([item[0] for item in kaggle_pkl])
        kaggle_label = pd.DataFrame(kaggle_label, columns=['emotion'])
        kaggle_label_dummy = np.array(pd.get_dummies(kaggle_label, columns=kaggle_label))
        kaggle_label_cls = np.argmax(np.array(kaggle_label_dummy, dtype=pd.Series), axis=1)
        kaggle_label_text = kaggle_label.emotion.tolist()
    return kaggle_img, kaggle_label_dummy, kaggle_label_cls, kaggle_label_text


# load data in the most recent format
def load_pd_data(file_path):
    with open(file_path, 'rb') as fin:
        images_pd = pickle.load(fin)
        images_pd = images_pd.dropna(axis=0, how='any')
        images_pd.reset_index(drop=True, inplace=True)
        img = np.asarray([images_pd.loc[i,'pixels'] for i in range(len(images_pd))])
        label_text = images_pd['emotion'].values.tolist()
        label_dummy = np.array(pd.get_dummies(label_text, columns=label_text))
        label_cls = np.argmax(np.array(label_dummy, dtype=pd.Series), axis=1)
    return img, label_dummy, label_cls, label_text


def next_batch(x_train, y_train, batch_size=128):
    num_images = len(x_train)
    id = np.random.choice(num_images, size=batch_size, replace=False)
    x_batch = x_train[id, :]
    y_batch = y_train[id, :]
    return x_batch, y_batch


def plot_images(images, img_shape, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9
    # Create figure with 5x5 sub-plots.
    fig, axes = plt.subplots(5, 5)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


def split_training_val(combined_size, combined_images, combined_labels):
    idx = np.random.permutation(combined_size)
    train_size = 0.8 * idx
    idx_train = idx[0:train_size]
    idx_validation = idx[train_size:]

    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]
    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    # Return the new training- and validation-sets.
    return x_train, y_train, x_validation, y_validation


starttime = time.asctime(time.localtime(time.time()))


def save_model(json_string, dirpath='../data/results/'):
    with open(dirpath + starttime + '.txt', 'w') as f:
        f.write(json_string)


def save_config(config, dirpath='../data/results/'):
    with open(dirpath + 'config_log.txt', 'a') as f:
        f.write(starttime + '\n')
        f.write(str(config) + '\n')


def save_result(train_val_accuracy, notes, conv_arch, dense, dirpath='../data/results/'):
    train_acc = train_val_accuracy['acc']
    val_acc = train_val_accuracy['val_acc']
    with open(dirpath + starttime + '_train_val.txt', 'w') as f:
        f.write(str(train_acc) + '\n')
        f.write(str(val_acc) + '\n')

    endtime = time.asctime(time.localtime(time.time()))
    with open(dirpath + 'result_log.txt', 'a') as f:
        f.write(starttime + '--' + endtime + ' comment: ' + notes + '\n')
        f.write(str(conv_arch) + ',' + str(dense) + '\n')
        f.write('Train acc: ' + str(train_acc[-1]) +
                'Val acc: ' + str(val_acc[-1]) +
                'Ratio: ' + str(val_acc[-1] / train_acc[-1]) + '\n')
