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
    else:
        raise ValueError("Invalid argument \"dataset\": %s" % args.dataset[0])

    #### Build Tensorflow graph #####

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

    #################################

    if os.path.exists(args.data_dir[0]):
        file_pairs = support.file_associations(args.data_dir[0])

    if not os.path.exists(args.output_dir[0]):
        sys.stdout.write("Directory \"%s\" does not exist. " % args.output_dir[0])
        sys.stdout.write("Do you want to create it? [y/N] ")
        sys.stdout.flush()
        user_input = sys.stdin.read(1)
        if user_input.lower()[0] != "y":
            sys.exit(0)

    os.makedirs(args.output_dir[0])

    sess = tf.Session()
    for split in file_pairs.keys():
        # Create directory for the split
        split_path = os.path.join(args.output_dir[0], split)
        os.mkdir(split_path)
        # Progress bar
        if show_progress:
            file_pairs_iter = tqdm(file_pairs[split], desc="%-7s" % split)
        else:
            file_pairs_iter = file_pairs[split]

        for file_pair in file_pairs_iter:
            # Extract properties from file_pair
            if file_pair[0].split(".")[-1] == "png":
                image_format = "png"
                image, image_shape = sess.run([file_contents, png_image_shape], \
                                              feed_dict={input_filename: file_pair[0]})
            elif file_pair[0].split(".")[-1] == "jpg" or \
                 file_pair[0].split(".")[-1] == "jpeg":
                image_format = "jpg"
                image, image_shape = sess.run([file_contents, jpg_image_shape], \
                                              feed_dict={input_filename: file_pair[0]})
            ext = file_pair[1].split(".")[-1]
            if ext != "png":
                raise ValueError("The label images need to be png files!" \
                                 "Got \"%s\"" % ext)
            label_format = "png"
            label, label_shape = sess.run([labels_encoding, png_image_shape], \
                                              feed_dict={input_filename: file_pair[1]})
            if image_shape[0] != label_shape[0] or \
               image_shape[1] != label_shape[1]:
                raise ValueError("Image dimensions does not match label.")
            # Note @image and @label are the raw png/jpg-encoded data
            features = { \
                "height":    _int64_feature(image_shape[0]), \
                "width":     _int64_feature(image_shape[1]), \
                "channels":  _int64_feature(image_shape[2]), \
                "image": _bytes_feature(image), \
                "label": _bytes_feature(label) \
            }
            tf_features = tf.train.Features(feature=features)
            tf_example  = tf.train.Example(features=tf_features)
            filename = os.path.split(file_pair[0])[-1].split(".")[0] + ".tfrecord"
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

