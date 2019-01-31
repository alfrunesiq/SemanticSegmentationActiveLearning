
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

def label_mapping(label_image):
    # TODO insert docstring here
    _, green, _ = tf.split(label_image, 3, axis=-1)
    green_int32 = tf.to_int32(green)
    return tf.nn.embedding_lookup(tf_green2trainId, green_int32)

def file_associations(root_path):
    #TODO add docstring
    image_dir = "rgb"
    label_dir = "GT_color"
    nir_dir   = "nir"

    _file_associations = {
        "train": {},
        "test":  {}
    }
    # _file_associations = {split: {id: {image/label/nir: path}}}
    for split in _file_associations.keys():
        # setup paths
        split_path = os.path.join(root_path, split)
        image_path = os.path.join(split_path, image_dir)
        label_path = os.path.join(split_path, label_dir)
        nir_path   = os.path.join(split_path, nir_dir)
        # create a dictionary of filename lookup tables due to the poor naming
        # of this dataset (in addition to the whole shape thing...)
        for filename in os.listdir(image_path):
            # Some _file_associations[split] has a "_Clipped" or "_mask" postfix
            image_id = filename.split(".")[0].split("_")[0]
            file_path = os.path.join(image_path, filename)
            _file_associations[split][image_id] = {}
            _file_associations[split][image_id]["image"] = file_path
        for filename in os.listdir(label_path):
            image_id = filename.split(".")[0].split("_")[0]
            file_path = os.path.join(label_path, filename)
            _file_associations[split][image_id]["label"] = file_path
        for filename in os.listdir(nir_path):
            image_id = filename.split(".")[0].split("_")[0]
            file_path = os.path.join(nir_path, filename)
            _file_associations[split][image_id]["nir"] = file_path

    return _file_associations
