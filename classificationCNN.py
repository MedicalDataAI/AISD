
import keras
import tensorflow as tf
import os
import random as rn
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
in_random_seed = 101
np.random.seed(in_random_seed)
rn.seed(in_random_seed)
# tf.set_random_seed(in_random_seed)
tf.random.set_seed(in_random_seed)  # tf2.0
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

def ConstructCNNBasedOnResnet50(in_input_size, in_weight, in_random_seed=101, in_lr2=0.01):
    pre_fun = keras.applications.resnet50.preprocess_input
    base_model = keras.applications.resnet50.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(in_input_size[0], in_input_size[1], 3))
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D(name='GlobalAverage')(x)
    x = keras.layers.Dense(1024,
                           activation='relu',
                           kernel_initializer=keras.initializers.glorot_normal(in_random_seed),
                           bias_initializer=keras.initializers.Zeros(),
                           kernel_regularizer=keras.regularizers.l2(in_lr2))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256,
                           activation='relu',
                           kernel_initializer=keras.initializers.glorot_normal(in_random_seed),
                           bias_initializer=keras.initializers.Zeros(),
                           kernel_regularizer=keras.regularizers.l2(in_lr2))(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(64,
                           activation='relu',
                           kernel_initializer=keras.initializers.glorot_normal(in_random_seed),
                           bias_initializer=keras.initializers.Zeros(),
                           kernel_regularizer=keras.regularizers.l2(in_lr2))(x)
    x = keras.layers.Dropout(0.5)(x)

    predictions = keras.layers.Dense(
        units=1,
        activation='sigmoid',
        kernel_initializer=keras.initializers.glorot_normal(in_random_seed),
        bias_initializer=keras.initializers.Zeros(),
        name='prediction')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=predictions, name="AISD")
    model.load_weights(in_weight)
    return model, pre_fun

def predictSingleImg(in_model, in_img_fp, in_img_size, pre_fun):
    img = keras.preprocessing.image.load_img(in_img_fp, target_size=(in_img_size[0], in_img_size[1]))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.squeeze(x)
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)
    elif len(x.shape) == 3:
        pass
    else:
        my_msg = r"Error in %s, the shape dim is %d." % (in_img_fp, len(x.shape))
        print(my_msg)
        exit(-1)
    x = np.expand_dims(x, axis=0)
    x = pre_fun(x)
    preds = in_model.predict(x)
    return preds
