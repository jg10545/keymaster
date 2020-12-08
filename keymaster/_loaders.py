# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def distort(x, outputshape=(128,128)):
    scale_x = tf.random.uniform((), minval=0.8, maxval=1.2)
    scale_y = tf.random.uniform((), minval=0.8, maxval=1.2)
    shear_x = tf.random.uniform((), minval=-0.3, maxval=0.3)
    shear_y = tf.random.uniform((), minval=-0.3, maxval=0.3)
    dxmax = x.shape[1]*scale_x/10
    dymax = x.shape[0]*scale_y/10
    dx = tf.random.uniform((), minval=-dxmax, maxval=dxmax)
    dy = tf.random.uniform((), minval=-dymax, maxval=dymax)

    tfm = tf.stack([scale_x, shear_x, dx, shear_y, scale_y, dy, 
                     tf.constant(0.0, dtype=tf.float32), 
                     tf.constant(0.0, dtype=tf.float32)], axis=0)
    tfm = tf.reshape(tfm, [1,8])
    distorted = tf.raw_ops.ImageProjectiveTransformV2(images=tf.expand_dims(x,0), transforms=tfm, 
                                      output_shape=outputshape, interpolation="BILINEAR")[0]
    return tf.reshape(distorted, (outputshape[0], outputshape[1], 3))


def distorted_pair_dataset(filepaths, num_parallel_calls=6, outputshape=(128,128), filetype="png"):
    
    def _load_and_distort(x):
        loaded = tf.io.read_file(x)
        if filetype == "png":
            decoded = tf.io.decode_png(loaded)
        elif filetype == "jpg":
            decoded = tf.io.decode_jpeg(loaded)
        else:
            assert False, "don't know this file type"
        resized = tf.image.resize(decoded, outputshape)
        cast = tf.cast(resized, tf.float32)/255
        return distort(cast, outputshape), distort(cast, outputshape)
    
    ds = tf.data.Dataset.from_tensor_slices(np.array(filepaths))
    ds = ds.shuffle(len(filepaths))
    ds = ds.map(_load_and_distort, num_parallel_calls=num_parallel_calls)
    return ds
