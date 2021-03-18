"""
Run inference over a test split
"""
import argparse
import json
import logging
import logging.config
import multiprocessing
import os
import sys

from tensorflow.compat import v1 as tf
tf.disable_eager_mode()

import matplotlib.pyplot as plt

import models
import datasets
import tensortools as tt

def decode_tfrecord(example, image_size):
    """
    Decoding clojure passed to the input pipeline.
    :param example: TFRecord example see dataset module for format details
    :returns: decoded image
    """
    image = tf.image.decode_image(example["image/data"],
                                 dtype=tf.float32,
                                 name="DecodeImage")
    image.set_shape(image_size)
    file_id = example["id"]
    return image, file_id

class PlotThread(object):
    def __init__(self, filepaths):
        self.idx = 0
        self.fig = plt.figure()
        self.ax  = self.fig.gca()
        self.img = None
        self.filepaths = filepaths

    def keyboard_callback(self, event):
        if event.key == "left":
            self.idx = (self.idx - 1) % len(self.filepaths)
        elif event.key == "right":
            self.idx = (self.idx + 1) % len(self.filepaths)
        self.img = plt.imread(self.filepaths[self.idx])
        self.ax.imshow(self.img)
        self.ax.set_xlabel(os.path.basename(self.filepaths[self.idx]))
        self.fig.canvas.draw()

    def __call__(self):
        # Wait for first image to be processed
        while len(self.filepaths) == 0:
            continue

        self.fig.canvas.mpl_connect("key_press_event", self.keyboard_callback)
        self.img = plt.imread(self.filepaths[self.idx])
        self.ax.imshow(self.img)
        self.ax.set_xlabel(os.path.basename(self.filepaths[self.idx]))
        plt.show()

def main(args):
    dataset = None
    # Retrieve datset
    if args.dataset == "cityscapes":
        dataset = datasets.Cityscapes()
    elif args.dataset == "freiburg":
        dataset = datasets.Freiburg()
    else:
        raise NotImplementedError("Dataset \"%s\" not yet supported."
                                  % args.dataset)
    # Configure directories
    data_dir = dataset.get_test_paths(args.data_dir)[0]
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # Parse first record and retrieve image dimensions
    first_record = os.path.join(data_dir, os.listdir(data_dir)[0])
    example = tt.tfrecord.tfrecord2example_dict(first_record)
    example = example["features"]["feature"]
    height   = example["height"]["int64List"]["value"][0]
    width    = example["width"]["int64List"]["value"][0]
    channels = example["image/channels"]["int64List"]["value"][0]
    decode_fn = lambda example: decode_tfrecord(example,
                                                [height, width, channels])

    # Create network and input stage
    net = models.ENet(dataset.num_classes)
    input_stage = tt.input.InputStage(input_shape=[height, width, channels])
    # Add test set to input stage
    num_examples = input_stage.add_dataset("test", data_dir, batch_size=1,
                                           decode_fn=decode_fn)

    input_image, file_id = input_stage.get_output()
    input_image = tf.expand_dims(input_image, axis=0)

    logits = net(input_image, training=False)
    p_class = tf.nn.softmax(logits)
    if args.size is not None:
        p_class = tf.image.resize_bilinear(logits, args.size)
    pred = tf.math.argmax(p_class, axis=-1)
    # Do the reverse embedding from trainId to dataset id
    if not args.color:
        pred = tf.expand_dims(pred, axis=-1)
        embedding  = tf.constant(dataset.embedding_reversed, dtype=tf.uint8)
        pred_embed = tf.gather_nd(embedding, pred)
        # Expand lost dimension
        pred_embed = tf.expand_dims(pred_embed, axis=-1)
    else:
        pred_embed = tf.gather(dataset.colormap, tf.cast(pred, tf.int32))
        pred_embed = tf.cast(pred_embed, tf.uint8)
    # Encode output image
    pred_encoding = tf.image.encode_png(pred_embed[0])

    # Write encoded file to @args.output_dir
    output_dir = args.output
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    filename = tf.string_join([file_id, ".png"])
    filepath = tf.string_join([output_dir, filename], separator="/")
    write_file = tf.io.write_file(filepath, pred_encoding)

    print("Loading checkpoint")
    # Restore model from checkpoint (@args.ckpt)
    ckpt = tf.train.Checkpoint(model=net)
    status = ckpt.restore(args.ckpt)
    print("Checkpoint loaded")
    if tf.__version__ < "1.14.0":
        status.assert_existing_objects_matched()
    else:
        status.expect_partial()

    # Create session and restore model
    sess = tf.Session()
    status.initialize_or_restore(sess)
    # Initialize input stage
    input_stage.init_iterator("test", sess)

    # Create visualization thread
    manager = multiprocessing.Manager()
    filepaths = manager.list()
    pt = PlotThread(filepaths)
    p = multiprocessing.Process(target=pt)
    p.start()
    # Loop over all images
    while True:
        try:
            _, _file_id, path = sess.run((write_file, file_id, filepath))
            filepaths.append(path.decode("ascii"))
            logger.info("Written processed sample %s" % str(_file_id))
        except tf.errors.OutOfRangeError:
            break
    logger.info("Inference successfully finished.")
    p.join()
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint",
                        type=str,
                        dest="ckpt", required=True,
                        help="Path to checkpoint file."
    )
    parser.add_argument("-d", "--data-dir",
                        type=str,
                        dest="data_dir", required=True,
                        help="Path to dataset test set directory."
    )
    parser.add_argument("-o", "--output",
                        type=str,
                        dest="output", required=True,
                        help="Output directory to store prediction map images."
    )
    parser.add_argument("-t", "--dataset",
                        type=str,
                        dest="dataset", required=True,
                        help="Dataset type: {cityscapes, freiburg}."
    )
    parser.add_argument("-s", "--output-size",
                        type=int,
                        nargs=2,
                        dest="size", required=False,
                        default=None,
                        help="Size of the output images."
    )
    parser.add_argument("--color",
                        action="store_true",
                        required=False,
                        default=False,
                        dest="color")
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    with open("util/logging.json") as conf:
        conf_dict = json.load(conf)
        logging.config.dictConfig(conf_dict)
        del conf_dict

    sys.exit(main(args))
