import argparse
import os

import numpy as np
import tensorflow as tf


def main(args):
    sess = tf.Session()
    for filename in os.listdir(args.data_dir[0]):
        img_path = os.path.join(args.data_dir[0], filename)
        lbl_path = os.path.join(args.label_dir[0], filename)
        image_data = tf.gfile.FastGFile(img_path, 'rb').read()
        image = tf.image.decode_png(img_path, channels=3)
        lbl_data = tf.gfile.FastGFile(lbl_path, 'rb').read()
        lbl = tf.image.decode_png(image_data, channels=3)
        example = tf.train.Example(features=tf.train.Features(feature={
            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
            "lbl": tf.train.Feature(bytes_list=tf.train.BytesList(value=[lbl_data]))
        }))
        fname_base = filename.split(".")[0]
        recName = os.path.join(args.output_dir[0], fname_base) + ".tfrecord"
        with tf.io.TFRecordWriter(recName) as f:
            f.write(example.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", \
                        type=str, \
                        nargs=1, \
                        dest="data_dir", \
                        required=True, \
                        help="Path to data set raw image data.", \
    )
    parser.add_argument("-l", "--label_dir", \
                        type=str, \
                        nargs=1, \
                        dest="label_dir", \
                        required=True, \
                        help="Path to data set label (ID) data.", \
    )
    parser.add_argument("-o", "--output_dir", \
                        type=str, \
                        nargs=1, \
                        dest="output_dir", \
                        required=True, \
                        help="Path to data set label (ID) data.", \
    )
    args = parser.parse_args()

    main(args)
