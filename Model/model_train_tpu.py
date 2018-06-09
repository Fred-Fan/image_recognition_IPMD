import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
import brewer2mpl
import pandas as pd
import pickle
#import cv2
#from Utils import load_pkl_data
#from Utils import load_pd_data
#from Utils import load_pd_direct

from keras.models import load_model

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import InputLayer, Input
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten,Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator
from tensorflow.contrib.tpu.python.tpu import tpu_optimizer


# Cloud TPU Cluster Resolvers
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string(
    "tpu_name", default=None,
    help="Name of the Cloud TPU for Cluster Resolvers. You must specify either "
    "this flag or --master.")

# Model specific paramenters
tf.flags.DEFINE_string(
    "master", default=None,
    help="GRPC URL of the master (e.g. grpc://ip.address.of.tpu:8470). You "
    "must specify either this flag or --tpu_name.")

tf.flags.DEFINE_integer("batch_size", 64,
                        "Mini-batch size for the computation. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.flags.DEFINE_string("train_file", "Data/", "Path to training data.")
tf.flags.DEFINE_integer("train_steps", 100000,
                        "Total number of steps. Note that the actual number of "
                        "steps is the next multiple of --iterations greater "
                        "than this value.")
tf.flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
tf.flags.DEFINE_string("model_dir", "Newmodel/", "Estimator model_dir")
tf.flags.DEFINE_integer("iterations_per_loop", 100,
                        "Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")


FLAGS = tf.flags.FLAGS



def model_fn(features, labels, mode, params):
    del params  # unused
    num_classes = 8

    x = Input(tensor=features)
    #x = InputLayer(input_shape=(img_size_flat,))
    #x = Reshape(img_shape)(x)

    # #model.add(Dropout(0.5, input_shape=(48, 48, 1)))
    x = Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
                activation='relu')(x)
    x = Conv2D(kernel_size=5, strides=1, filters=32, padding='same',
              activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
# 
    x = Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                activation='relu')(x)
    x = Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                activation='relu')(x)
    x = Conv2D(kernel_size=10, strides=1, filters=64, padding='same',
                activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
              activation='relu')(x)
    x = Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
                activation='relu')(x)
    x = Conv2D(kernel_size=15, strides=1, filters=128, padding='same',
                activation='relu')(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    # # Last fully-connected / dense layer with softmax-activation
    # # for use in classification.
    logits = Dense(num_classes)(x)

    loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels
                )
            )
    optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)

    if FLAGS.use_tpu:
        optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tpu_estimator.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions={
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        )
def input_fn(params):
    """Read CIFAR input data from a TFRecord dataset."""
    del params

    img_size = 96
    image_channel = 1
    img_size_flat = img_size * img_size * image_channel
    img_shape = [image_channel, img_size, img_size]

    emotion_dict = {'Anger':0, 'Contempt':1,
                'Disgust':2, 'Fear':3,
                'Happy':4, 'Neutral':5,
                'Sad':6, 'Surprise':7}
    input_file = os.path.join(FLAGS.train_file, 'mtrain96.pd')
    with open(input_file, 'rb') as fin:
        temp = pickle.load(fin)

    features = np.array([list(x) for x in temp["pixels"]])
    labels = np.array([emotion_dict[x] for x in temp["emotion"]])

    assert features.shape[0] == labels.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    images, labels = dataset.make_one_shot_iterator().get_next()
    return images, labels

'''
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
'''

def main(argv):
    del argv
    tf.logging.set_verbosity(tf.logging.INFO)


    #if FLAGS.master is None and FLAGS.tpu_name is None:
    #    raise RuntimeError("You must specify either --master or --tpu_name.")
#
    #if FLAGS.master is not None:
    #    if FLAGS.tpu_name is not None:
    #        tf.logging.warn("Both --master and --tpu_name are set. Ignoring "
    #                  "--tpu_name and using --master.")
    #    tpu_grpc_url = FLAGS.master
    #else:
    #    tpu_cluster_resolver = (
    #        tf.contrib.cluster_resolver.TPUClusterResolver(
    #            FLAGS.tpu_name,
    #            zone=FLAGS.tpu_zone,
    #            project=FLAGS.gcp_project))
    #    tpu_grpc_url = tpu_cluster_resolver.get_master()

    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=FLAGS.model_dir,
        save_checkpoints_secs=3600,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tf.contrib.tpu.TPUConfig(FLAGS.iterations_per_loop, FLAGS.num_shards))

    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=FLAGS.batch_size,
        eval_batch_size=FLAGS.batch_size,
        params={"data_dir": FLAGS.train_file},
        config=run_config)

    estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)
'''
    run_config = tpu_config.RunConfig(
        master=tpu_grpc_url,
        model_dir=FLAGS.model_dir,
        save_checkpoints_secs=3600,
        session_config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True),
        tpu_config=tpu_config.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_shards),
        )

    estimator = tpu_estimator.TPUEstimator(
        model_fn=model_fn,
        use_tpu=FLAGS.use_tpu,
        config=run_config,
        train_batch_size=FLAGS.batch_size)
    estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)
'''


if __name__ == "__main__":
    # tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)


'''
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5, \
                          verbose=1, mode='auto')
callbacks_list = [earlystop]

epochs = 50
batch_size = 256
train_X, train_Y, _, _, train_file = load_pd_data('Data/mtrain96.pd')
test_X, test_Y, _, _, test_file = load_pd_data('Data/mtest96.pd')
#model = build_cnn_model()
model = load_model("Orimodel/mymodel.h5py")
optimizer = Adam(lr = 1e-4)
if FLAGS.use_tpu:
    optimizer = tpu_optimizer.CrossShardOptimizer(optimizer)
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
'''
