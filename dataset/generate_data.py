"""
This script creates a set of tensorflow records for each image- label- pair.

python generate_data.py --data_set {kitti,cityscapes,freiburg} \
                        --root_dir /path/to/dataset/root
"""
import os
import sys

import argparse

import tensorflow as tf

show_progress = False
try:
    from tqdm import tqdm
    show_progress = True
except ImportError:
    pass

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def main(args):
    if args.dataset[0].lower() == "cityscapes":
        import support.cityscapes as support
    elif args.dataset[0].lower() == "freiburg":
        import support.freiburg as support
    else:
        raise ValueError("Invalid argument \"dataset\": %s" % args.dataset[0])

    ################ Build Tensorflow Graph ##################
    input_filename = tf.placeholder(dtype=tf.string)
    file_contents  = tf.read_file(input_filename)
    # Seperate heads for decoding png or jpg
    jpg_decoding = tf.image.decode_jpeg(file_contents)
    png_decoding = tf.image.decode_png(file_contents)
    # Remapping of labels (can only be png)
    labels_mapped   = support.label_mapping(png_decoding)
    labels_encoding = tf.image.encode_png(labels_mapped)
    # Get the shape of image / labels to assert them equal
    png_image_shape = tf.shape(png_decoding)
    jpg_image_shape = tf.shape(jpg_decoding)
    ##########################################################

    if os.path.exists(args.data_dir[0]):
        dataset_paths = support.file_associations(args.data_dir[0])

    if not os.path.exists(args.output_dir[0]):
        sys.stdout.write("Directory \"%s\" does not exist. " % args.output_dir[0])
        sys.stdout.write("Do you want to create it? [y/N] ")
        sys.stdout.flush()
        user_input = sys.stdin.read(1)
        if user_input.lower()[0] != "y":
            sys.exit(0)
        else:
            os.makedirs(args.output_dir[0])

    sess = tf.Session()
    for split in dataset_paths.keys():
        # Create directory for the split
        split_path = os.path.join(args.output_dir[0], split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)
        # Progress bar
        if show_progress:
            example_iter = tqdm(list(dataset_paths[split].items()),
                                desc="%-7s" % split)
        else:
            example_iter = list(dataset_paths[split].items())
        # TODO process multiple files in parallel (in proportion to #CPU)
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
                        fetches=[labels_encoding, png_image_shape],
                        feed_dict={input_filename: path})
                    features["label"] = _bytes_feature(label)
                else: # image data
                    # Check file extension
                    if ext == "png":
                        image, shape = sess.run(
                            fetches=[file_contents, png_image_shape],
                            feed_dict={input_filename: path})
                    elif ext == "jpg" or ext == "jpeg":
                        ext = "jpg"
                        image, shape = sess.run(
                            fetches=[file_contents, jpg_image_shape],
                            feed_dict={input_filename: path})
                    else: # TODO add support for tiff by wrapping in tf.py_func
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
            # Construct feature example
            tf_features = tf.train.Features(feature=features)
            tf_example  = tf.train.Example(features=tf_features)
            filename = example[0] + ".tfrecord"
            with tf.gfile.Open(os.path.join(split_path, filename), 'wb') as f:
                f.write(tf_example.SerializeToString())

if __name__ == "__main__":
    # Setup commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root", \
                        type=str, \
                        nargs=1, \
                        dest="data_dir", \
                        required=True, \
                        help="Path to data set root directory.")
    parser.add_argument("-t", "--dataset", \
                        type=str, \
                        nargs=1, \
                        dest="dataset", \
                        required=True, \
                        help="Name of the dataset {cityscapes,freiburg,kitti}.")
    parser.add_argument("-o", "--output_dir", \
                        type=str, \
                        nargs=1, \
                        dest="output_dir", \
                        required=True, \
                        help="Path to where to store the records.")
    args = parser.parse_args()
    main(args)
