=========
keymaster
=========

.. image:: docs/keymaster.png

Tensorflow 2 implementation of *Unsupervised Learning of Object Landmarks through Conditional Image Generation* by Jakab *et al*. 

Currently in a very, very rough prototype state.

Paper here: https://arxiv.org/abs/1806.07823

Original code repo: https://github.com/tomasjakab/imm

So far I've tested on pairs of faces (with random scale/shear distortions) from the Flickr faces thumbnails dataset. Example TensorBoard output (from left to right- input x, input y, y with keypoints, and y reconstructed from image x and keypoints from y):

.. image:: docs/tensorboard1.png

* Free software: MIT license


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
