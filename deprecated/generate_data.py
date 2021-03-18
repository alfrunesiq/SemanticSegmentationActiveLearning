"""
This script creates a set of tensorflow records for each image- label- pair.

python generate_data.py --data_set {kitti,cityscapes,freiburg} \
                        --root_dir /path/to/dataset/root
"""
import os
import sys

import argparse

import numpy as np
from tensorflow.compat import v1 as tf

import datasets

show_progress = False
try:
    from tqdm import tqdm
    show_progress = True
except ImportError:
    pass
try:
    import cv2
except ImportError:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  _bytes = value if not (isinstance(value, str) and sys.version_info[0] == 3) \
                 else value.encode()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[_bytes]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main(args):
    dataset = None
    scale_factor = args.scale_factor
    scale_factor_image = None
    scale_factor_label = None

    if args.dataset.lower() == "cityscapes":
        use_coarse = True if args.extra is not None and \
                             "coarse" in args.extra[0].lower() \
                     else False
        dataset = datasets.Cityscapes(use_coarse)
    elif args.dataset.lower() == "freiburg":
        modalities = None if args.extra is None else args.extra
        dataset = datasets.Freiburg(modalities)
    elif args.dataset.lower() == "vistas":
        dataset = datasets.Vistas()
    else:
        raise ValueError("Invalid argument \"dataset\": %s" % args.dataset)

    ################ Build Tensorflow Graph ##################
    input_filename = tf.placeholder(dtype=tf.string)
    file_contents  = tf.read_file(input_filename)
    # Seperate heads for decoding png or jpg
    image_decoding = tf.image.decode_image(file_contents)
    label_decoding = tf.image.decode_image(file_contents)
    # Get the shape of image / labels to assert them equal
    image_shape = tf.shape(image_decoding)
    label_shape = tf.shape(label_decoding)
    if args.width is not None:
        scale_factor_image = image_shape[1] / args.width
        scale_factor_label = label_shape[1] / args.width
    if scale_factor_image is not None:
        scale_factors_image = tf.stack([scale_factor_image,
                                        scale_factor_image, 1])
        scale_factors_label = tf.stack([scale_factor_label,
                                        scale_factor_label, 1])
        # Compute rescaled shapes
        image_shape = tf.cast(tf.round(tf.cast(image_shape, tf.float64)
                                       / scale_factors_image), tf.int32)
        label_shape = tf.cast(tf.round(tf.cast(label_shape, tf.float64)
                                       / scale_factors_label), tf.int32)

        image_decoding = tf.image.resize_nearest_neighbor(
            tf.expand_dims(image_decoding, axis=0), image_shape[:-1])
        label_decoding = tf.image.resize_nearest_neighbor(
            tf.expand_dims(label_decoding, axis=0), label_shape[:-1])
        image_decoding = tf.squeeze(image_decoding, axis=0)
        label_decoding = tf.squeeze(label_decoding, axis=0)
    image_encoding = tf.cond(
        tf.strings.regex_full_match(input_filename, ".+\.png$"),
        true_fn =lambda: tf.image.encode_png(image_decoding),
        false_fn=lambda: tf.image.encode_jpeg(image_decoding))
    # Remapping of labels (can only be png)
    embedding = tf.constant(dataset.embedding, dtype=tf.uint8)
    label_remapped = tf.gather_nd(embedding, tf.cast(label_decoding, tf.int32))
    label_remapped = tf.expand_dims(label_remapped, axis=-1)
    label_encoding = tf.image.encode_png(label_remapped)
    # In order to convert tiff to png
    tif_input_image = tf.placeholder(tf.uint8, shape=[None,None,None])
    tif_png_encoding = tf.image.encode_png(tif_input_image)
    ##########################################################

    if os.path.exists(args.data_dir):
        dataset_paths = dataset.file_associations(args.data_dir)
    else:
        raise ValueError("Dataset path does not exist\n%s\n" % args.data_dir)

    if not os.path.exists(args.output_dir):
        sys.stdout.write("Directory \"%s\" does not exist. "
                         % args.output_dir)
        sys.stdout.write("Do you want to create it? [y/N] ")
        sys.stdout.flush()
        user_input = sys.stdin.read(1)
        if user_input.lower()[0] != "y":
            sys.exit(0)
        else:
            os.makedirs(args.output_dir)

    # Create session on CPU
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = ""
    sess = tf.Session(config=config)
    # Write records for each split
    for split in dataset_paths.keys():
        # Create directory for the split
        split_path = os.path.join(args.output_dir, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)
        # Progress bar
        if show_progress:
            example_iter = tqdm(list(dataset_paths[split].items()),
                                desc="%-7s" % split,
                                ascii=True,
                                dynamic_ncols=True)
        else:
            example_iter = list(dataset_paths[split].items())
        # Iterate over all examples in split and gather samples in
        # separate records
        for example in example_iter:
            # example = [str(ID), dict({str(type): str(path)})]
            features = {}
            shapes   = []
            for _type in example[1].keys():
                # Only "label" key need to be treated differently all other
                # is assumed to contain image data (rgb/nir/depthmap)
                path = example[1][_type]
                ext  = path.split(".")[-1] # path extension
                if "label" in _type: # label data
                    # Check file extension
                    if ext != "png":
                        raise ValueError(
                            "The label images need to be png files!"
                            "Got \"%s\"" % ext)
                    label, shape = sess.run(
                        fetches=[label_encoding, label_shape],
                        feed_dict={input_filename: path})
                    features["label"] = _bytes_feature(label)
                else: # image data
                    # Handle the different file extensions separately
                    if ext == "tif" or ext == "tiff":
                        # read image and convert to png
                        ext = "png"
                        # Read image as is (iscolor=-1)
                        image = cv2.imread(path, -1)
                        shape = image.shape
                        if len(shape) == 3 and shape[-1] == 3:
                            # Opencv defaults to BGR whereas Tensorflow RGB
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        elif len(shape) == 2:
                            image = np.expand_dims(image, axis=-1)
                        image = sess.run(tif_png_encoding,
                                         feed_dict={tif_input_image: image})
                    elif ext == "png" or ext == "jpg" or ext == "jpeg":
                        image, shape = sess.run(
                            fetches=[image_encoding, image_shape],
                            feed_dict={input_filename: path})
                    else:
                        raise ValueError(
                            "Unsupported image format \"%s\"" % ext)
                    if len(shape) == 3:
                        channels = shape[2]
                    else:
                        channels = 1
                    # note that @_type/data is the raw image encoding
                    features[_type + "/channels"] = _int64_feature(channels)
                    features[_type + "/data"]     = _bytes_feature(image)
                    features[_type + "/encoding"] = _bytes_feature(ext)
                shapes.append(shape)
            # END for _type in example[1].keys()
            # Check that shapes are consistent
            for i in range(1,len(shapes)):
                if shapes[i][0] != shapes[i-1][0] or \
                   shapes[i][1] != shapes[i-1][1]:
                    raise ValueError(
                        "Image dimensions does not match label.\n"
                        "Got: %s" % shapes)
            # Add shape info to feature. Note that channels are allready added
            # and label image is assumed to be single channel png image.
            features["height"] = _int64_feature(shape[0])
            features["width"]  = _int64_feature(shape[1])
            features["id"]     = _bytes_feature(example[0])
            # Construct feature example
            tf_features = tf.train.Features(feature=features)
            tf_example  = tf.train.Example(features=tf_features)
            filename = example[0] + ".tfrecord"
            with tf.io.TFRecordWriter(
                    os.path.join(split_path, filename)) as f:
                f.write(tf_example.SerializeToString())
    # Write feature keys in order to dynamically being able to reconstruct the
    # content of the records when reading the records.
    meta_file = os.path.join(args.output_dir, "meta.txt")
    with open(meta_file, "w") as f:
        f.write("\n".join(features.keys()))
    # In order to reconstruct:
    # features = {}
    # with open(meta_file, "r") as f:
    #   ln = f.readline()
    #   if ln.endswith("/data") or \
    #      ln.endswith("/encoding") or \
    #      ln == "label":
    #      features[ln] = tf.FixedLenFeature([], tf.string)
    #   else:
    #      features[ln] = tf.FixedLenFeature([], tf.int64)
    #
    # .../channels -> tf.FixedLenFeature([], tf.int64),
    # .../data     -> tf.FixedLenFeature([], tf.string),
    # .../encoding -> tf.FixedLenFeature([], tf.string),
    # label        -> tf.FixedLenFeature([], tf.string),
    # height       -> tf.FixedLenFeature([], tf.int64),
    # width        -> tf.FixedLenFeature([], tf.int64),


if __name__ == "__main__":
    # Setup commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root",
                        type=str,
                        dest="data_dir",
                        required=True,
                        help="Path to data set root directory.")
    parser.add_argument("-t", "--dataset",
                        type=str,
                        dest="dataset",
                        required=True,
                        help="Name of the dataset {cityscapes,freiburg,kitti}.")
    parser.add_argument("-o", "--output_dir",
                        type=str,
                        dest="output_dir",
                        required=True,
                        help="Path to where to store the records.")
    parser.add_argument("-s", "--scale",
                        type=float,
                        default=1,
                        dest="scale_factor",
                        required=False,
                        help="Downscaling factor.")
    parser.add_argument("-w", "--width",
                        type=int,
                        default=None,
                        dest="width",
                        required=False,
                        help="Downscaling factor.")
    # TODO: make this a little nicer
    parser.add_argument(
        "-e", "--extra", type=str, nargs="*",
        dest="extra", required=False,
        help="Extra arguments. This is dependent on the particular dataset "
        "selected with the \"-t\" flag.\n"
        "%-10s : {gtCoarse} - coarsely annotated dataset.\n"
        "%-10s : {nir,nir_gray,depth_gray,(etc)} - additional modalities."
        % ("cityscapes", "freiburg"))
    args = parser.parse_args()
    main(args)
