# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os

from keymaster._models import build_encoder, build_decoder, _vgg_filter_model
from keymaster._loaders import distorted_pair_dataset
from keymaster._keypoint import generate_gaussians

def _perceptual_loss(filter_model, orig, recon, weights=[100.0, 1.6, 2.3, 1.8, 2.8, 100.0]):
    print("using different weights for perceptual loss")
    weights=[10.0, 1, 1, 1, 1, 1]
    orig_features = filter_model(orig)
    recon_features = filter_model(recon)
    loss = 0
    for o,r,w in zip(orig_features, recon_features, weights):
        loss += w*tf.reduce_mean((o-r)**2)
    return loss


def _build_training_step(encoder, poseencoder, decoder, optimizer, 
                         loss="perceptual", vgg_filter_model=None):
    @tf.function
    def trainstep(x,y, loss="perceptual"):
        print("tracing training step")
        allvars = encoder.trainable_variables+poseencoder.trainable_variables+decoder.trainable_variables
        with tf.GradientTape() as tape:
            features = encoder(x, training=True)
            pose_features = poseencoder(y, training=True)
            pose_predictions, pose_means = generate_gaussians(pose_features, 0.1)
            reconstruction = decoder(tf.concat([features, pose_predictions], 3), training=True)
        
            # reconstruction_loss
            if loss == "reconstruction":
                loss = tf.reduce_mean((y-reconstruction)**2)
            elif loss == "perceptual":
                loss = _perceptual_loss(vgg_filter_model, y, reconstruction)
        
        grads = tape.gradient(loss, allvars)
        optimizer.apply_gradients(zip(grads, allvars))
        return loss
    return trainstep




class Trainer(object):
    """
    """
    
    
    def __init__(self, K, logdir, trainingdata, imshape=(128,128),
                 batch_size=64, num_parallel_calls=6, losstype="perceptual",
                 lr=1e-3):
        """
        :logdir: (string) path to log directory
        
        """
        self.logdir = logdir
        
        if logdir is not None:
            self._file_writer = tf.summary.create_file_writer(logdir, flush_millis=10000)
            self._file_writer.set_as_default()
        self.step = 0
        
        ds = distorted_pair_dataset(trainingdata, outputshape=imshape)
        ds = ds.batch(batch_size, drop_remainder=True)
        self.ds = ds.prefetch(1)
        
        encoder = build_encoder()
        poseencoder = build_encoder(poseencoder=K)
        decoder = build_decoder(K)
        self.optimizer = tf.keras.optimizers.Adam(lr)
        if losstype == "perceptual":
            vgg_filter_model = _vgg_filter_model()
        else:
            vgg_filter_model = None
        self.trainstep = _build_training_step(encoder, poseencoder, decoder,
                                              optimizer=self.optimizer,
                                              loss=losstype, 
                                              vgg_filter_model=vgg_filter_model)
        self.models = {"encoder":encoder, "decoder":decoder, 
                       "poseencoder":poseencoder}
        
        
        # generate a fixed batch for comparison in tensorboard
        for x,y in self.ds:
            self._batchx = x
            self._batchy = y
            break
        self._visualize_outputs()

    
    def fit(self, epochs=1, save=True):
        """

        """
        for e in tqdm(range(epochs)):
            for x,y in self.ds:
                loss = self.trainstep(x,y)
                self._record_scalars(loss=loss)
                self.step += 1
            
            if save:
                self.save()
            self._visualize_outputs()
    
    def save(self):
        """
        Write model(s) to disk
        
        Note: tried to use SavedModel format for this and got a memory leak;
        think it's related to https://github.com/tensorflow/tensorflow/issues/32234
        
        For now sticking with HDF5
        """
        for m in self.models:
            path = os.path.join(self.logdir, m+".h5")
            self.models[m].save(path, overwrite=True, save_format="h5")
            
    def _visualize_outputs(self):
        features = self.models["encoder"](self._batchx)
        pose_features = self.models["poseencoder"](self._batchy)
        pose_predictions, pose_means = generate_gaussians(pose_features, 0.1)
        reconstruction = self.models["decoder"](tf.concat([features, pose_predictions], 3))
        alltogether = tf.concat([self._batchx, self._batchy, reconstruction], 2)
        self._record_images(reconstruction=alltogether)
            
    def _record_scalars(self, metric=False, **scalars):
        for s in scalars:
            tf.summary.scalar(s, scalars[s], step=self.step)
            
            if metric:
                if hasattr(self, "_mlflow"):
                    self._log_metrics(scalars, step=self.step)
            
    def _record_images(self, **images):
        for i in images:
            tf.summary.image(i, images[i], step=self.step, max_outputs=10)
            
    def _record_hists(self, **hists):
        for h in hists:
            tf.summary.histogram(h, hists[h], step=self.step)
            
        
    def _get_current_learning_rate(self):
        # return the current value of the learning rate
        # CONSTANT LR CASE
        if isinstance(self._optimizer.lr, tf.Variable) or isinstance(self._optimizer.lr, tf.Tensor):
            return self._optimizer.lr
        # LR SCHEDULE CASE
        else:
            return self._optimizer.lr(self.step)
            
        
    def __del__(self):
        if hasattr(self, "_mlflow"):
            import mlflow
            mlflow.end_run()
