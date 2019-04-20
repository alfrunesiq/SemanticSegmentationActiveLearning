#!/usr/bin/python
# NOTE: This code is adopted from the official cityscapesScripts repository.
#       http://github.com/mcordts/cityscapesScripts
#       cityscapesScripts/helpers/labels.py
#
# Cityscapes labels
#
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

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#-------------------------------------------------------------------------------
# A list of all labels
#-------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id   trainId   category         catId   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,     255 , 'void'         , 0     , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,     255 , 'void'         , 0     , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,     255 , 'void'         , 0     , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,     255 , 'void'         , 0     , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,     255 , 'void'         , 0     , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,     255 , 'void'         , 0     , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,     255 , 'void'         , 0     , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,       0 , 'flat'         , 1     , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,       1 , 'flat'         , 1     , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,     255 , 'flat'         , 1     , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,     255 , 'flat'         , 1     , True         , (230,150,140) ),
    Label(  'building'             , 11 ,       2 , 'construction' , 2     , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,       3 , 'construction' , 2     , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,       4 , 'construction' , 2     , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,     255 , 'construction' , 2     , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,     255 , 'construction' , 2     , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,     255 , 'construction' , 2     , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,       5 , 'object'       , 3     , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,     255 , 'object'       , 3     , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,       6 , 'object'       , 3     , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,       7 , 'object'       , 3     , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,       8 , 'nature'       , 4     , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,       9 , 'nature'       , 4     , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,      10 , 'sky'          , 5     , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,      11 , 'human'        , 6     , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,      12 , 'human'        , 6     , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,      13 , 'vehicle'      , 7     , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,      14 , 'vehicle'      , 7     , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,      15 , 'vehicle'      , 7     , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,     255 , 'vehicle'      , 7     , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,     255 , 'vehicle'      , 7     , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,      16 , 'vehicle'      , 7     , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,      17 , 'vehicle'      , 7     , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,      18 , 'vehicle'      , 7     , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,     255 , 'vehicle'      , 7     , True         , (  0,  0,142) ),
]

#-------------------------------------------------------------------------------
# Support functions - ID mappings and file association list
#-------------------------------------------------------------------------------
class Cityscapes:
    def __init__(self, coarse=False):

        # Embedding from label image to train id
        # NOTE: defer creation untill usage to avoid memory usage
        self._embedding = None
        self._embedding_reversed = None
        self._name_embedding = None
        self._colormap = None

        # Number of training classes
        self._num_classes = 19

        # Use coarse annotation set
        self.coarse = coarse

    @property
    def colormap(self):
        if self._colormap is None:
            self._colormap = np.full((256,3), 255, dtype=np.uint8)
            for label in reversed(labels):
                self._colormap[label.trainId] = label.color
        return self._colormap

    @property
    def embedding(self):
        if self._embedding is None:
            self._embedding = np.full(256, 255, dtype=np.uint8)
            for label in reversed(labels):
                self._embedding[label.id] = label.trainId
        return self._embedding

    @property
    def name_embedding(self):
        if self._name_embedding is None:
            self._name_embedding = []
            for label in labels:
                if label.trainId != 255:
                    self._name_embedding.append(label.name)
        return self._name_embedding

    @property
    def embedding_reversed(self):
        if self._embedding_reversed is None:
            self._embedding_reversed = np.zeros(256, dtype=np.uint8)
            for label in reversed(labels):
                self._embedding_reversed[label.trainId] = label.id
        return self._embedding_reversed

    @property
    def num_classes(self):
        return self._num_classes

    def get_train_paths(self, root_path):
        paths = [os.path.join(root_path, "train")]
        if self.coarse:
            paths.append(os.path.join(root_path, "train_extra"))
        return paths

    def get_validation_paths(self, root_path):
        return [os.path.join(root_path, "val")]

    def get_test_paths(self, root_path):
        return [os.path.join(root_path, "test")]

    def file_associations(self, root_path):
        """
        Returns a dictionary of file associations between raw image data and
        semantic labels.
        File tree structure:

        root
        ├── camera
        │   └── split
        │       └── city
        │           └── camera intrinsics and extrinsics
        └── image_type {leftImg8bit,gtFine,gtCoarse}
            └── split
                └── city
                    └── city_sequenceid_frameid_imagetype.ext

        :param root_path: path to dataset root directory
        :returns: dictionary of file associations for each split
        :rtype:   dict{str: list(tuple(str,str))}
        """
        # The dataset is organized using following filename paths
        #{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
        splits = ["train", "val"]
        label_type = "gtCoarse" if self.coarse else "gtFine"
        image_type = "leftImg8bit"
        image_path_base = os.path.join(root_path, image_type)
        label_path_base = os.path.join(root_path, label_type)
        _file_associations = {
            "train": {},
            "val":   {},
            "test":  {}
        }
        if self.coarse:
            _file_associations["train_extra"] = {}
        else:
            _file_associations["test"] = {}

        # Iterate over file tree and associate image and label paths
        for split in splits:
            # Update path to {split} scope
            image_path_split = os.path.join(image_path_base, split)
            label_path_split = os.path.join(label_path_base, split)
            for city in os.listdir(label_path_split):
                # Update path to {city} scope
                image_path_city = os.path.join(image_path_split, city)
                label_path_city = os.path.join(label_path_split, city)
                # TODO this could be vectorized
                for filename in os.listdir(label_path_city):
                    label_id = filename.split("_")
                    # file_id = [city, seq, frame, type, ext]
                    # filter out instance seg. labels and polygon description files
                    if label_id[-1] != "labelIds.png":
                        continue
                    file_id = "_".join(label_id[:3])
                    # Construct the corresponding raw image filename
                    image_id = label_id[:-1]
                    image_id[-1] = (image_type + ".png")
                    image_name = "_".join(image_id)
                    # Construct file association entry
                    image_path = os.path.join(image_path_city, image_name)
                    label_path = os.path.join(label_path_city, filename)
                    _file_associations[split][file_id] = {}
                    _file_associations[split][file_id]["image"] = image_path
                    _file_associations[split][file_id]["label"] = label_path

        # Treat test samples separately as they don't contain any labels
        image_path_split = os.path.join(image_path_base, split)
        for root, dirs, filenames in os.walk(image_path_split):
            for filename in filenames:
                label_id = filename.split("_")
                file_id = "_".join(label_id[:3])

                image_path = os.path.join(root, filename)
                _file_associations["test"][file_id] = {}
                _file_associations["test"][file_id]["image"] = image_path

        return _file_associations
