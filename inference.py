"""
Run inference over a test split
"""
import argparse
import json
import logging
import logging.config
import os
import sys

import tensorflow as tf

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

def main(args):
    dataset = None
    if args.dataset == "cityscapes":
        dataset = datasets.Cityscapes()
    elif args.dataset == "freiburg":
        dataset = datasets.Freiburg()
    else:
        raise NotImplementedError("Dataset \"%s\" not yet supported."
                                  % args.dataset)
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

    net = models.ENet(dataset.num_classes)
    input_stage = tt.input.InputStage(1, input_shape=[height, width, channels])
    num_examples = input_stage.add_dataset("test", data_dir,
                                           decode_fn=decode_fn)

    input_image, file_id = input_stage.get_output("test")
    input_image = tf.expand_dims(input_image, axis=0)

    logits = tf.squeeze(net(input_image, training=False), axis=0)
    pred   = tf.math.argmax(logits, axis=-1)
    embedding  = tf.constant(dataset.embedding_reversed, dtype=tf.uint8)
    pred_embed = tf.gather(embedding, pred, axis=0)
    pred_embed = tf.expand_dims(pred_embed, axis=-1)
    pred_encoding = tf.image.encode_png(pred_embed)

    filename = tf.string_join([file_id, ".png"])
    output_dir = args.output
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]
    filepath = tf.string_join([output_dir, filename], separator="/")
    write_file = tf.io.write_file(filepath, pred_encoding)

    ckpt = tf.train.Checkpoint(model=net)
    status = ckpt.restore(args.ckpt)
    status.assert_existing_objects_matched()

    sess = tf.Session()
    status.initialize_or_restore(sess)
    input_stage.init_iterator("test", sess)

    while True:
        try:
            _, _file_id = sess.run((write_file, file_id))
            logger.info("Written processed sample %s" % str(_file_id))
        except tf.errors.OutOfRangeError:
            break

    logger.info("Inference successfully finished.")
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
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    with open("util/logging.json") as conf:
        conf_dict = json.load(conf)
        logging.config.dictConfig(conf_dict)
        del conf_dict

    sys.exit(main(args))
