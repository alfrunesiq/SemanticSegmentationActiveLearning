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

    'hasInstances', # Whether this label distinguishes between single instances or not

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
    #       name                     id   trainId   category         catId   hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,     255 , 'void'         , 0     , False        , True         , (  0,  0,  0) ), \
    Label(  'ego vehicle'          ,  1 ,     255 , 'void'         , 0     , False        , True         , (  0,  0,  0) ), \
    Label(  'rectification border' ,  2 ,     255 , 'void'         , 0     , False        , True         , (  0,  0,  0) ), \
    Label(  'out of roi'           ,  3 ,     255 , 'void'         , 0     , False        , True         , (  0,  0,  0) ), \
    Label(  'static'               ,  4 ,     255 , 'void'         , 0     , False        , True         , (  0,  0,  0) ), \
    Label(  'dynamic'              ,  5 ,     255 , 'void'         , 0     , False        , True         , (111, 74,  0) ), \
    Label(  'ground'               ,  6 ,     255 , 'void'         , 0     , False        , True         , ( 81,  0, 81) ), \
    Label(  'road'                 ,  7 ,       0 , 'flat'         , 1     , False        , False        , (128, 64,128) ), \
    Label(  'sidewalk'             ,  8 ,       1 , 'flat'         , 1     , False        , False        , (244, 35,232) ), \
    Label(  'parking'              ,  9 ,     255 , 'flat'         , 1     , False        , True         , (250,170,160) ), \
    Label(  'rail track'           , 10 ,     255 , 'flat'         , 1     , False        , True         , (230,150,140) ), \
    Label(  'building'             , 11 ,       2 , 'construction' , 2     , False        , False        , ( 70, 70, 70) ), \
    Label(  'wall'                 , 12 ,       3 , 'construction' , 2     , False        , False        , (102,102,156) ), \
    Label(  'fence'                , 13 ,       4 , 'construction' , 2     , False        , False        , (190,153,153) ), \
    Label(  'guard rail'           , 14 ,     255 , 'construction' , 2     , False        , True         , (180,165,180) ), \
    Label(  'bridge'               , 15 ,     255 , 'construction' , 2     , False        , True         , (150,100,100) ), \
    Label(  'tunnel'               , 16 ,     255 , 'construction' , 2     , False        , True         , (150,120, 90) ), \
    Label(  'pole'                 , 17 ,       5 , 'object'       , 3     , False        , False        , (153,153,153) ), \
    Label(  'polegroup'            , 18 ,     255 , 'object'       , 3     , False        , True         , (153,153,153) ), \
    Label(  'traffic light'        , 19 ,       6 , 'object'       , 3     , False        , False        , (250,170, 30) ), \
    Label(  'traffic sign'         , 20 ,       7 , 'object'       , 3     , False        , False        , (220,220,  0) ), \
    Label(  'vegetation'           , 21 ,       8 , 'nature'       , 4     , False        , False        , (107,142, 35) ), \
    Label(  'terrain'              , 22 ,       9 , 'nature'       , 4     , False        , False        , (152,251,152) ), \
    Label(  'sky'                  , 23 ,      10 , 'sky'          , 5     , False        , False        , ( 70,130,180) ), \
    Label(  'person'               , 24 ,      11 , 'human'        , 6     , True         , False        , (220, 20, 60) ), \
    Label(  'rider'                , 25 ,      12 , 'human'        , 6     , True         , False        , (255,  0,  0) ), \
    Label(  'car'                  , 26 ,      13 , 'vehicle'      , 7     , True         , False        , (  0,  0,142) ), \
    Label(  'truck'                , 27 ,      14 , 'vehicle'      , 7     , True         , False        , (  0,  0, 70) ), \
    Label(  'bus'                  , 28 ,      15 , 'vehicle'      , 7     , True         , False        , (  0, 60,100) ), \
    Label(  'caravan'              , 29 ,     255 , 'vehicle'      , 7     , True         , True         , (  0,  0, 90) ), \
    Label(  'trailer'              , 30 ,     255 , 'vehicle'      , 7     , True         , True         , (  0,  0,110) ), \
    Label(  'train'                , 31 ,      16 , 'vehicle'      , 7     , True         , False        , (  0, 80,100) ), \
    Label(  'motorcycle'           , 32 ,      17 , 'vehicle'      , 7     , True         , False        , (  0,  0,230) ), \
    Label(  'bicycle'              , 33 ,      18 , 'vehicle'      , 7     , True         , False        , (119, 11, 32) ), \
    Label(  'license plate'        , -1 ,     255 , 'vehicle'      , 7     , False        , True         , (  0,  0,142) ), \
] # NOTE last Label is augmented in this script


#-------------------------------------------------------------------------------
# Create embedding list for trainIds
#-------------------------------------------------------------------------------
id2trainId = [label.trainId for label in labels]

id2trainId_tf = tf.constant(id2trainId, dtype=tf.uint8)
#-------------------------------------------------------------------------------
# Support functions - ID mappings and file association list
#-------------------------------------------------------------------------------
def label_mapping(label_image):
    """
    Employs the lookup table @id2trainId for converting dataset label id to
    training label indices.
    NOTE: labels that should be ignored are mapped to the value 255
          remember to mask these pixels when evaluating the loss

    :param label_image: Segmentation ground truth image
    :returns: The converted image as a tensorflow tensor.
    :rtype:   tf.Tensor
    """
    labels_int32 = tf.to_int32(label_image)
    return tf.nn.embedding_lookup(id2trainId_tf, labels_int32)

def file_associations(root_path, coarse=False):
    """
    Returns a dictionary of file associations between raw image data and
    semantic labels.

    :param root_path: path to dataset root directory
    :param coarse:    whether to use the coarsely annotated dataset
    :returns: dictionary of file associations for each split
    :rtype:   dict{str: list(tuple(str,str))}
    """
    # The dataset is organized using following filename paths
    #{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
    label_type = "gtCoarse" if coarse else "gtFine"
    image_type = "leftImg8bit"
    image_path_base = os.path.join(root_path, image_type)
    label_path_base = os.path.join(root_path, label_type)
    filepairs = {
        "train": [], \
        "val": [], \
        "test": []
    }
    if coarse:
        filepairs["train_extra"] = []

    for split in filepairs.keys():
        # Update path to {split} scope
        image_path_split = os.path.join(image_path_base, split)
        label_path_split = os.path.join(label_path_base, split)
        for city in os.listdir(label_path_split):
            # Update path to {city} scope
            image_path_city = os.path.join(image_path_split, city)
            label_path_city = os.path.join(label_path_split, city)
            for filename in os.listdir(label_path_city):
                file_id = filename.split("_")
                # file_id = [city, seq, frame, type, ext]
                # filter out instance seg. labels and polygon description files
                if file_id[-1] != "labelIds.png":
                    continue
                # Construct the corresponding raw image filename
                image_id = file_id[:-1]
                image_id[-1] = (image_type + ".png")
                imagename = "_".join(image_id)
                # Construct filepair
                filepair = (os.path.join(image_path_city, imagename), \
                            os.path.join(label_path_city, filename))
                # Add pair to global dictionary
                filepairs[split].append(filepair)
    return filepairs
