# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
import holoviews as hv
import panel as pn
import colorcet
import json
from sklearn import linear_model

from keymaster._keypoint import get_keypoints

def _load(filepath, size=(128,128)):
    """
    PIL wrapper to load an image into a numpy array
    """
    return np.array(Image.open(filepath).resize(size, resample=Image.BILINEAR)).astype(np.float32)/255



def _init_empty_labels(categories, imshape=(128,128)):
    x = list(np.random.randint(0, imshape[1], len(categories)))
    y = list(np.random.randint(0, imshape[0], len(categories)))
    color = colorcet.glasbey_light[:len(categories)]
    return {"x":x, "y":y, "category":[c for c in categories], "color":color}

def _build_hv_labeler(imfile, imshape=(128,128), annos=None):
    opts = {"default_tools":[]}
    img_hv = hv.RGB(_load(imfile, imshape), bounds=(0,0,imshape[0], imshape[1])).opts(**opts)
    if annos is None:
        annos = {"x":[], "y":[], "color":[]}
    opts = {"tools":["hover"], "default_tools":[], "color":"color", "size":40,
           "line_color":"black", "padding":0.1}
    points = hv.Points(annos, vdims=["color", "category"]).opts(**opts)
    points_annotator = hv.annotate.instance()
    
    #opts = {"active_tools":["point_annotator_tool"]}
    compose = hv.annotate.compose(img_hv, points_annotator(points, annotations={"category":str}))#.opts(**opts)
    return compose, points_annotator


class KeyPointLabeler(object):
    """
    Interactive widget for trying to learn task-specific keypoints
    from unsupervised keypoints.
    
    KeyPointLabeler.panel contains the panel GUI object.
    """
    
    def __init__(self, filepaths, categories, imshape=(128,128), outfile=None, 
                 poseencoder=None, labels=[]):
        """
        :filepaths: list of strings; paths to images
        :categories: list of strings; label for each task-specific keypoint
        :imshape: shape to resize images to
        :poseencoder: keras model that generates unsupervised keypoints
        :outfile: optional; path to JSON file to store your labels
        :labels: optional; previous labels to start from
        """
        self.categories = categories
        self._imshape = imshape
        self._filepaths = filepaths
        # make a copy
        self._unused = [f for f in filepaths]
        self._outfile = outfile
        self._poseencoder = poseencoder
        self._color =  colorcet.glasbey_light[:len(self.categories)]
        
        self._labels = labels
        self._unsup_keypoints = {}
        
        # initialize GUI
        self._currentfile = self._choose_new_image()
        c,p = _build_hv_labeler(self._currentfile, imshape=imshape, annos=_init_empty_labels(categories))
        self._chooser = pn.pane.HoloViews(c)
        self._pointsannotator = p
        buttons = {"save":pn.widgets.Button(name="save and continue", button_type="primary"),
                   "discard":pn.widgets.Button(name="discard and continue", button_type="danger"),
                   "train":pn.widgets.Button(name="train", button_type="success")}
        self._buttons = buttons
        self._buttons["save"].on_click(self._save_and_continue_callback)
        self._buttons["discard"].on_click(self._discard_and_continue_callback)
        self._buttons["train"].on_click(self.train)
        self._messagebar = pn.pane.Markdown(" ", width=600)
        self.panel = pn.Column(self._chooser,
                              pn.Row(buttons["save"], buttons["discard"], buttons["train"]),
                              self._messagebar)
        
    def save_json(self):
        if self._outfile is not None:
            json.dump(self._labels, open(self._outfile, "w"))
            
    def fit(self, **kwargs):
        pass
    
    def _choose_new_image(self):
        return self._unused.pop(np.random.choice(np.arange(len(self._unused))))
    
    def sample(self):
        newfile = self._choose_new_image()
        self._currentfile = newfile
        
        if hasattr(self, "_xmodel"):
            #assert False, "not yet implemented"
            annos = self.predict(newfile, as_annos=True)
        else:
            annos = _init_empty_labels(self.categories)
        c,p = _build_hv_labeler(self._choose_new_image(), imshape=self._imshape,
                                annos=annos)
        self._chooser.object = c
        self._pointsannotator = p
        self._messagebar.object = "%s labels"%len(self._labels)
        
    def record_current_labels(self):
        df = self._pointsannotator.annotated.dframe()
        j = {"filepath":self._currentfile, 
             "annotations":{"x":[float(x) for x in df.x.values/self._imshape[1]], 
             "y":[float(x) for x in df.y.values/self._imshape[0]],
             "category":list(df.category.values)}}
        self._labels.append(j)
        self.save_json()
        
    def _save_and_continue_callback(self, *events):
        self.record_current_labels()
        self.sample()
        
    def _discard_and_continue_callback(self, *events):
        self._unused.append(self._currentfile)
        self.sample()
        
    def _add_feature(self, f):
        if f not in self._unsup_keypoints:
            self._unsup_keypoints[f] = get_keypoints(f, 
                                                self._poseencoder, size=self._imshape)
    
    def predict(self, f, as_annos=False):
        H,W = self._imshape
        self._add_feature(f)
        x = W*np.maximum(0,
                       np.minimum(self._xmodel.predict(self._unsup_keypoints[f][:,1].reshape(1,-1)),W)).ravel()
        y = H*np.maximum(0,
                       np.minimum(self._ymodel.predict(self._unsup_keypoints[f][:,0].reshape(1,-1)),H)).ravel()
        if as_annos:
            return {"x":x, "y":y, "category":self.categories, "color":self._color}
        else:
            return x,y
        
    
    def train(self, *events):
        if self._poseencoder is None:
            self._messagebar.object = "## NEED A POSE ENCODER FOR THIS FOOL"
            return False
        if len(self._labels) == 0:
            self._messagebar.object = "## NEED LABELS FIRST FOOL"
            return False
        # update unsup keypoint "features" for every labeled file
        self._messagebar.object = "updating unsupervised keypoint features"
        for l in self._labels:
            self._add_feature(l["filepath"])
            
        
        # build X and Y datasets
        self._messagebar.object = "assembling features for training"
        X_labels = np.stack([np.array(x["annotations"]["x"]) for x in self._labels], 0)
        Y_labels = np.stack([np.array(x["annotations"]["y"]) for x in self._labels], 0)
        X_covariates = np.stack([self._unsup_keypoints[f["filepath"]][:,1] for f in self._labels],0)
        Y_covariates = np.stack([self._unsup_keypoints[f["filepath"]][:,0] for f in self._labels],0)
        # build and fit models
        self._messagebar.object = "training"
        if X_labels.shape[0] >= 10:
            self._xmodel = linear_model.MultiTaskElasticNetCV()
            self._ymodel = linear_model.MultiTaskElasticNetCV()
        else:
            self._xmodel = linear_model.ElasticNet()
            self._ymodel = linear_model.ElasticNet()
            
        self._xmodel.fit(X_covariates, X_labels)
        self._ymodel.fit(Y_covariates, Y_labels)
        self._messagebar.object = "done"     
        
