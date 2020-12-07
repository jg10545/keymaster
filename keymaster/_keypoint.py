# -*- coding: utf-8 -*-
import tensorflow as tf


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
