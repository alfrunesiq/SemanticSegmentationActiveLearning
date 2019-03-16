from collections import namedtuple

import os

import numpy as np

Label = namedtuple( "Label", [
    "name"         ,
    "id"           ,
    "trainId"      ,
    "cityscapesId" ,
    "color"        ,
])

labels = [
    #       name                       id   trainId  cityscapesId   color
    Label( "bird"                    ,  0 ,     255 ,           5 , (165,  42,  42) ),
    Label( "ground animal"           ,  1 ,     255 ,           5 , (  0, 192,   0) ),
    Label( "curb"                    ,  2 ,       1 ,           8 , (196, 196, 196) ),
    Label( "fence"                   ,  3 ,       4 ,          13 , (190, 153, 153) ),
    Label( "guard rail"              ,  4 ,     255 ,          14 , (180, 165, 180) ),
    Label( "barrier"                 ,  5 ,     255 ,         255 , ( 90, 120, 150) ),
    Label( "wall"                    ,  6 ,       3 ,          12 , (102, 102, 156) ),
    Label( "bike lane"               ,  7 ,       1 ,           8 , (128,  64, 255) ),
    Label( "crosswalk - plain"       ,  8 ,     255 ,         255 , (140, 140, 200) ),
    Label( "curb cut"                ,  9 ,       1 ,           8 , (170, 170, 170) ),
    Label( "parking"                 , 10 ,     255 ,           9 , (250, 170, 160) ),
    Label( "pedestrian area"         , 11 ,       1 ,           7 , ( 96,  96,  96) ),
    Label( "rail track"              , 12 ,     255 ,          10 , (230, 150, 140) ),
    Label( "road"                    , 13 ,       0 ,           7 , (128,  64, 128) ),
    Label( "service lane"            , 14 ,     255 ,         255 , (110, 110, 110) ),
    Label( "sidewalk"                , 15 ,       1 ,           8 , (244,  35, 232) ),
    Label( "bridge"                  , 16 ,     255 ,          15 , (150, 100, 100) ),
    Label( "building"                , 17 ,       2 ,          11 , ( 70,  70,  70) ),
    Label( "tunnel"                  , 18 ,     255 ,          16 , (150, 120,  90) ),
    Label( "person"                  , 19 ,      11 ,          24 , (220,  20,  60) ),
    Label( "bicyclist"               , 20 ,      12 ,          25 , (255,   0,   0) ),
    Label( "motorcyclist"            , 21 ,      12 ,          25 , (255,   0, 100) ),
    Label( "other rider"             , 22 ,      12 ,          25 , (255,   0, 200) ),
    Label( "lane marking - crosswalk", 23 ,     255 ,         255 , (200, 128, 128) ),
    Label( "lane marking - general"  , 24 ,       0 ,           7 , (255, 255, 255) ),
    Label( "mountain"                , 25 ,     255 ,           4 , ( 64, 170,  64) ),
    Label( "sand"                    , 26 ,     255 ,         255 , (230, 160,  50) ),
    Label( "sky"                     , 27 ,      10 ,          23 , ( 70, 130, 180) ),
    Label( "snow"                    , 28 ,     255 ,           5 , (190, 255, 255) ),
    Label( "terrain"                 , 29 ,     255 ,          22 , (152, 251, 152) ),
    Label( "vegetation"              , 30 ,       8 ,          21 , (107, 142,  35) ),
    Label( "water"                   , 31 ,     255 ,           4 , (  0, 170,  30) ),
    Label( "banner"                  , 32 ,     255 ,         255 , (255, 255, 128) ),
    Label( "bench"                   , 33 ,     255 ,           4 , (250,   0,  30) ),
    Label( "bike rack"               , 34 ,     255 ,         255 , (100, 140, 180) ),
    Label( "billboard"               , 35 ,     255 ,           4 , (220, 220, 220) ),
    Label( "catch basin"             , 36 ,     255 ,         255 , (220, 128, 128) ),
    Label( "cctv camera"             , 37 ,     255 ,           4 , (222,  40,  40) ),
    Label( "fire hydrant"            , 38 ,     255 ,           4 , (100, 170,  30) ),
    Label( "junction box"            , 39 ,     255 ,           4 , ( 40,  40,  40) ),
    Label( "mailbox"                 , 40 ,     255 ,           4 , ( 33,  33,  33) ),
    Label( "manhole"                 , 41 ,     255 ,         255 , (100, 128, 160) ),
    Label( "phone booth"             , 42 ,     255 ,           4 , (142,   0,   0) ),
    Label( "pothole"                 , 43 ,     255 ,         255 , ( 70, 100, 150) ),
    Label( "street light"            , 44 ,     255 ,           0 , (210, 170, 100) ),
    Label( "pole"                    , 45 ,       5 ,          17 , (153, 153, 153) ),
    Label( "traffic sign frame"      , 46 ,     255 ,         255 , (128, 128, 128) ),
    Label( "utility pole"            , 47 ,       5 ,          17 , (  0,   0,  80) ),
    Label( "traffic light"           , 48 ,       6 ,          19 , (250, 170,  30) ),
    Label( "traffic sign (back)"     , 49 ,     255 ,         255 , (192, 192, 192) ),
    Label( "traffic sign (front)"    , 50 ,       7 ,          20 , (220, 220,   0) ),
    Label( "trash can"               , 51 ,     255 ,           4 , (140, 140,  20) ),
    Label( "bicycle"                 , 52 ,      18 ,          33 , (119,  11,  32) ),
    Label( "boat"                    , 53 ,     255 ,         255 , (150,   0, 255) ),
    Label( "bus"                     , 54 ,      15 ,          28 , (  0,  60, 100) ),
    Label( "car"                     , 55 ,      13 ,          26 , (  0,   0, 142) ),
    Label( "caravan"                 , 56 ,     255 ,          29 , (  0,   0,  90) ),
    Label( "motorcycle"              , 57 ,      17 ,          32 , (  0,   0, 230) ),
    Label( "on rails"                , 58 ,      16 ,          31 , (  0,  80, 100) ),
    Label( "other vehicle"           , 59 ,     255 ,         255 , (128,  64,  64) ),
    Label( "trailer"                 , 60 ,     255 ,          30 , (  0,   0, 110) ),
    Label( "truck"                   , 61 ,      14 ,          27 , (  0,   0,  70) ),
    Label( "wheeled slow"            , 62 ,     255 ,         255 , (  0,   0, 192) ),
    Label( "car mount"               , 63 ,     255 ,         255 , ( 32,  32,  32) ),
    Label( "ego vehicle"             , 64 ,     255 ,           1 , (120,  10,  10) ),
    Label( "unlabeled"               , 65 ,     255 ,           0 , (  0,   0,   0) )
]

class Vistas:
    def __init__(self):

        # Embedding from label image to train id
        self._embedding = None
        self._colormap  = np.ones((256,3), dtype=np.uint8)*255
        for label in labels:
            self._colormap[label.trainId] = label.color

        # Number of training classes
        self._num_classes = 19

    @property
    def colormap(self):
        if self._colormap is None:
            self._colormap  = np.ones((256,3), dtype=np.uint8)*255
            for label in labels:
                self._colormap[label.trainId] = label.color
        return self._colormap

    @property
    def embedding(self):
        # Due to 4M size defer creation until requested
        if self._embedding is None:
            self._embedding = np.ones((256,256,256), dtype=np.uint8)*255
            for label in labels:
                self._embedding[label.color] = label.trainId

        return self._embedding

    @property
    def embedding_reversed(self):
        if self._colormap is None:
            self._colormap  = np.ones((256,3), dtype=np.uint8)*255
            for label in labels:
                self._colormap[label.trainId] = label.color
        return self._colormap

    @property
    def num_classes(self):
        return self._num_classes

    def get_train_paths(self, root_path):
        paths = [os.path.join(root_path, "train")]
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
        └── split
            └── data_type
                └── files_id.{png,jpg}

        :param root_path: path to dataset root directory
        :returns: dictionary of file associations for each split
        :rtype:   dict{str: list(tuple(str,str))}
        """
        # The dataset is organized using following filename paths
        #{root}/{type}{video}/{split}/{city}/{city}_{seq:0>6}_{frame:0>6}_{type}{ext}
        splits = ["training", "validation", "testing"]
        data_types = ["images", "labels"]
        name_map = {
            "training"   : "train",
            "validation" : "val",
            "testing"    : "test",
            "images"     : "image",
            "labels"     : "label"
        }
        _file_associations = {
            "train": {},
            "val":   {},
            "test":   {}
        }

        # Iterate over file tree and associate image and label paths
        for root, dirs, filenames in os.walk(root_path):
            basename = os.path.basename(root)
            if basename in splits:
                _split = name_map[basename]
                continue
            if basename not in data_types:
                continue
            for filename in filenames:
                dtype = name_map[basename]
                file_id = "".join(filename.split(".")[:-1])
                _filename = os.path.join(root, filename)
                _file_associations[_split].setdefault(file_id,{})[dtype] = _filename
        return _file_associations
