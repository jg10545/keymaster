# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


def _get_coord(x, other_axis, axis_size):
    ## x is output of the final pose encoder conv layer- should be B,H,W,NMAP
    # get "x-y" coordinates:
    g_c_prob = tf.reduce_mean(x, axis=other_axis)  # B,W,NMAP
    g_c_prob = tf.nn.softmax(g_c_prob, axis=1)  # B,W,NMAP
    #coord_pt = tf.to_float(tf.linspace(-1.0, 1.0, axis_size)) # W
    coord_pt = tf.linspace(-1.0, 1.0, axis_size) # W
    coord_pt = tf.reshape(coord_pt, [1, axis_size, 1]) ## 1,W,1
    g_c = tf.reduce_sum(g_c_prob * coord_pt, axis=1) # compute expectation along axis, scaled by coord_pt
    return g_c, g_c_prob


def _get_gaussian_maps(mu, shape_hw, inv_std):
    """
    based on https://github.com/tomasjakab/imm/blob/0fee6b24466a5657d66099694f98036c3279b245/imm/models/imm_model.py#L34
  
    Generates [B,SHAPE_H,SHAPE_W,NMAPS] tensor of 2D gaussians,
    given the gaussian centers: MU [B, NMAPS, 2] tensor.
    STD: is the fixed standard dev.
    """
    mu_y, mu_x = mu[:, :, 0:1], mu[:, :, 1:2]
    y = tf.linspace(-1.0, 1.0, shape_hw[0])

    x = tf.linspace(-1.0, 1.0, shape_hw[1])



    #elif mode == 'ankush':
    y = tf.reshape(y, [1, 1, shape_hw[0]])
    x = tf.reshape(x, [1, 1, shape_hw[1]])

    g_y = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_y - y) * inv_std)))
    g_x = tf.exp(-tf.sqrt(1e-4 + tf.abs((mu_x - x) * inv_std)))

    g_y = tf.expand_dims(g_y, axis=3)
    g_x = tf.expand_dims(g_x, axis=2)
    g_yx = tf.matmul(g_y, g_x)  # [B, NMAPS, H, W]

    g_yx = tf.transpose(g_yx, perm=[0, 2, 3, 1])
    return g_yx


def generate_gaussians(pose_maps, std=0.1):
    gauss_y, gauss_y_probs = _get_coord(pose_maps, 2, pose_maps.shape[1])
    gauss_x, gauss_x_probs = _get_coord(pose_maps, 1, pose_maps.shape[2])
    gauss_mu = tf.stack([gauss_y, gauss_x], axis=2)
    gauss_xy_ = _get_gaussian_maps(gauss_mu, pose_maps.shape[1:3], 1.0 /std)
    return gauss_xy_, gauss_mu



def _load(filepath, size=(128,128)):
    """
    PIL wrapper to load an image into a numpy array
    """
    return np.array(Image.open(filepath).resize(size, resample=Image.BILINEAR)).astype(np.float32)/255



def get_keypoints(img, poseencoder, plot=False, size=(128,128)):
    """
    Get keypoints for an image
    
    :img: filepath to image, or float numpy array containing image (normalized to unit interval)
    :poseencoder: keras model for pose encoder
    :plot: whether to plot results
    :size: if loading image, size to reshape to
    """
    # make sure image is loaded, resized, and padded with a batch dimension
    if isinstance(img, str):
        img = _load(img, size)
    if len(img.shape) == 3:
        img = np.expand_dims(img, 0)
    # compute keypoint feature and then build gaussian centers from them
    W = img.shape[1]
    keypoint_features = poseencoder.predict(img)
    _, gaussians = generate_gaussians(keypoint_features)
    gaussians = gaussians.numpy()[0]
    # rescale to image pixel coords
    gaussians = (W/2)*gaussians + W/2
    if plot:
        plt.imshow(img[0])
        plt.plot(gaussians[:,1], gaussians[:,0], "o")
        
    return gaussians