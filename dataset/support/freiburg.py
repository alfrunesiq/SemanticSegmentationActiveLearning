
from __future__ import print_function, absolute_import, division
from collections import namedtuple

import os

import numpy as np
import tensorflow as tf

#-------------------------------------------------------------------------------
# Definitions
#-------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [
    'name'    , # label name
    'id'      , # id in the dataset "README"
    'trainId' , # label training id
    'color'   , # label color in ground truth images
    ] )

labels = [
    #     name            id    trainId    color (R,G,B)
    Label("Void"       ,   0 ,  255     ,           None  ),
    Label("Road"       ,   1 ,    0     , (170, 170, 170) ),
    Label("Grass"      ,   2 ,    1     , (  0, 255,   0) ),
    Label("Vegetation" ,   3 ,    2     , (102, 102,  51) ),
    Label("Tree"       ,   4 ,    3     , (  0,  60,   0) ),
    Label("Sky"        ,   5 ,    4     , (  0, 120, 255) ),
    Label("Obstacle"   ,   6 ,    5     , (  0,   0,   0) ),
]

#-------------------------------------------------------------------------------
# Create embedding look-up-table for green channel
#-------------------------------------------------------------------------------
green2trainId = np.ones(256, dtype=np.uint8)*255
for label in labels[1:]:
    green2trainId[label.color[1]] = label.trainId
tf_green2trainId = tf.constant(green2trainId, dtype=tf.uint8)

class Freiburg:
    def __init__(self, modalities=None):
        self.modalities = modalities

    def label_mapping(self, label_image):
        # TODO insert docstring here
        _, green, _ = tf.split(label_image, 3, axis=-1)
        green_int32 = tf.to_int32(green)
        return tf.nn.embedding_lookup(tf_green2trainId, green_int32)

    def get_label_mapping(self):
        return green2trainId

    def file_associations(self, root_path):
        #TODO add docstring
        image_dir = "rgb"
        label_dir = "GT_color"
        nir_dir   = "nir_gray"
        if self.modalities == None:
            self.modalities = ["rgb"]
        elif not isinstance(self.modalities, list):
            raise ValueError(
                "ERROR: Modalities need to be a list of strings "
                "containing the name of modalities as in the dataset "
                "filetree, e.g. {rgb,nir_gray,...}.")

        _file_associations = {
            "train": {},
            "test":  {}
        }
        # _file_associations = {split: {id: {image/label/nir: path}}}
        for split in _file_associations.keys():
            # setup paths
            split_path = os.path.join(root_path, split)
            label_path = os.path.join(split_path, label_dir)
            # create a dictionary of filename lookup tables and mind the poor
            # naming convension of this dataset
            for filename in os.listdir(label_path):
                # Some _file_associations[split] has a "_Clipped" or "_mask" postfix
                _id = filename.split(".")[0].split("_")[0]
                file_path = os.path.join(label_path, filename)
                _file_associations[split][_id] = {}
                _file_associations[split][_id]["label"] = file_path
            for modality in self.modalities:
                mod_path = os.path.join(split_path, modality)
                if modality == "rgb":
                    modality = "image" # to be consistent with other datasets
                for filename in os.listdir(mod_path):
                    _id = filename.split(".")[0].split("_")[0]
                    file_path = os.path.join(mod_path, filename)
                    _file_associations[split][_id][modality] = file_path
        return _file_associations
