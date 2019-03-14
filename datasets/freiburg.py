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
    #     name            id    trainId   color (R,G,B)
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

class Freiburg:
    def __init__(self, modalities=None):
        # Embedding for mapping GT_color (green) to trainId
        self._embedding = None#np.ones(256, dtype=np.uint8)*255
        self._colormap  = None#np.ones((256, 3), dtype=np.uint8)*255
        self._embedding_reversed = self._colormap

        self.modalities = modalities
        self._num_classes = 6

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = np.full((256,256,256), 255, dtype=np.uint8)
            for label in labels[1:]:
                self._embedding[label.color] = label.trainId
        return self._embedding

    @property
    def embedding_reversed(self):
        if self._colormap is None:
            self._colormap = np.full((255, 3), 255, dtype=np.uint8)
            for label in labels[1:]:
                self._colormap[label.trainId] = label.color
        return self._colormap

    @property
    def colormap(self):
        if self._colormap is None:
            self._colormap = np.full((255, 3), 255, dtype=np.uint8)
            for label in labels[1:]:
                self._colormap[label.trainId] = label.color
        return self._colormap

    @property
    def num_classes(self):
        return self._num_classes

    def get_train_paths(self, root_path):
        return [os.path.join(root_path, "train")]

    def get_validation_paths(self, root_path):
        val_path = [os.path.join(root_path, "val")]
        if os.path.exists(val_path):
            return val_path
        else:
            return

    def get_test_paths(self, root_path):
        return [os.path.join(root_path, "test")]

    def label_mapping(self, label_image):
        """
        Maps the rgb label image to the respective ids given in dataset
        README file.
        :param label_image: tf.Tensor with the label image
        :returns: The remapped label image
        :rtype:   tf.Tensor (single channel image)

        """
        # TODO insert docstring here
        embedding_tf = tf.constant(self.embedding, dtype=tf.uint8)
        _, green, _ = tf.split(label_image, 3, axis=-1)
        green_int32 = tf.cast(green, dtype=tf.int32)
        return tf.nn.embedding_lookup(tf_green2trainId, green_int32)

    def file_associations(self, root_path, val_proportion=0.05):
        """
        Creates a dictionary of file associations, due to the awkward
        naming convension used for this dataset the code below looks a
        bit messy.
        :param root_path:      dataset root path
        :param val_proportion: proportion of training images moved to
                               validation set
        :returns: dictionary with fileassociations
        :rtype:   dict {"split": {"id": {"type": "path"}}}
                  type in e.g. "image" / "label" / "nir"
        """
        image_dir = "rgb"
        label_dir = "GT_color"
        if self.modalities == None:
            self.modalities = ["rgb"]
        elif not isinstance(self.modalities, list):
            raise ValueError(
                "ERROR: Modalities need to be a list of strings "
                "containing the name of modalities as in the dataset "
                "filetree, e.g. {rgb,nir_gray,...}.")
        _file_associations = {
            "train": {},
            "test" : {}
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

        # Move some training examples to validation set
        if val_proportion > 0.0:
            _file_associations["val"] = {}
            # Calculate stride used for uniformly sampling the training ids
            stride = int((1.0 / val_proportion + 1))

            # Sample uniformly from the sorted list of training ids
            examples = [example for example in sorted(
                _file_associations["train"])[::stride]]
            # Move samples from train dict to val dict
            for example in examples:
                _file_associations["val"][example] = \
                    _file_associations["train"].pop(example)

        return _file_associations
