# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf



def build_encoder(filters=32, batchnorm=True, poseencoder=False):
    """
    poseencoder: number of keypoint layers if this is the pose encoder; otherwise False
    """
    inpt = tf.keras.layers.Input((None, None, 3))
    net = inpt
    
    num_filters = np.array([1,1,2,2,4,4,8,8])*filters
    kernel_sizes = [7,3,3,3,3,3,3,3]
    strides = [1,1,2,1,2,1,2,1]
    for n,k,s in zip(num_filters, kernel_sizes, strides):
        net = tf.keras.layers.Conv2D(n,k,strides=s, padding="same")(net)
        if batchnorm:
            net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        
    if poseencoder:
        net = tf.keras.layers.Conv2D(poseencoder, 1, 1)(net)
    return tf.keras.Model(inpt, net)


def build_decoder(K, filters=32, batchnorm=True):
    inpt = tf.keras.layers.Input((None, None, K+8*filters))
    net = inpt
    
    num_filters = np.array([8,4,2,1])*filters
    for e,n in enumerate(num_filters):
        if e > 0:
            net = tf.keras.layers.UpSampling2D(size=(2,2), interpolation="bilinear")(net)
        net = tf.keras.layers.Conv2D(n, 3, padding="same")(net)
        if batchnorm: net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Conv2D(n, 3, padding="same")(net)
        if batchnorm: net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)
        net = tf.keras.layers.Conv2D(n, 3, activation="relu", padding="same")(net)
        
    print("note- using sigmoid instead of linear outputs from paper")
    net = tf.keras.layers.Conv2D(3, 3, padding="same", activation="sigmoid")(net)
    return tf.keras.Model(inpt, net)



def _vgg_filter_model():
    """
    Return a VGG16 model that outputs all the layers used for perceptual loss
    """
    vgg = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
    outlayers = [0, 2, 5, 8, 12, 16]
    return tf.keras.Model(vgg.layers[0].input, [vgg.layers[i].output for i in outlayers])