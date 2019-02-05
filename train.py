# Python standard libraries
import argparse
import json
import logging
import logging.config
import os
import sys

# Non-standard includes
import tensorflow as tf

# User includes
import models
from tensortools import InputStage, losses

def main(args):
    # TODO insert train loop and network initialization here
    #      BTW: use tf.train.Saver class to store checkpoints
    # Setup input pipeline
    data_paths = []
    classes = 0
    if args["dataset"] == "cityscapes":
        classes = 19
        data_paths.append(os.path.join(args["data_dir"], "train"))
        if args["coarse"]:
            data_paths.append(os.path.join(args["data_dir"], "train_extra"))

    elif args["dataset"] == "freiburg":
        classes = 6
        data_paths.append(os.path.join(args["data_dir"], "train"))

    input = InputStage(args["batch_size"], args["size"])
    input.add_dataset("train", data_paths, epochs=args["epochs"],augment=True)


    with tf.Session() as sess:
        sess = tf.Session()
        image, label = input.output("train", sess)

        net = models.ENet(classes, True)
        pred, params = net.build(image)

        # Create checkpoint saver object
        saver = tf.train.Saver(var_list=net.get_vars())
        # Create summary writer object
        summary_writer = tf.summary.FileWriter(args["log_dir"])
        # TODO setup loss, summaries etc
        loss = losses.masked_softmax_cross_entropy(label,
                                                   net.get_logits(),
                                                   classes)
        summary = tf.summary.scalar("Loss", loss)
        optimizer = tf.train.AdamOptimizer(args["learning_rate"])
        train_op = optimizer.minimize(loss)
        sess.run(tf.global_variables_initializer())
        # MAIN training loop
        while True:
            try:
                _, summary_serialized, pred, raw = sess.run([train_op, summary, pred, (image, label)])
                summary_writer.add_summary(summary_serialized)
            except tf.errors.OutOfRangeError:
                break

    # TODO remove below
    import matplotlib.pyplot as plt
    import numpy as np
    for i in range(args["batch_size"]):
        plt.figure().gca().imshow(np.squeeze(raw[0][i,:,:,:3]))
        plt.figure().gca().imshow(np.squeeze(raw[1][i]))
        plt.figure().gca().imshow(np.squeeze(pred[i]))
    plt.show()
    # TODO Remove downto here
    return 0

class HelpfullParser(argparse.ArgumentParser):
    # Prints help instead of usage string
    def error(self, message):
        self.print_help()
        self.exit(2, "error: %s\n" % message)

def parse_arguments():
    """
    Handles parseing of commandline arguments and loading of defeaults
    from `parameters.json`.

    :returns: The parsed commandline options
    :rtype:   dict
    """
    # Load default arguments
    default = None
    with open("parameters.json") as f:
        default = json.load(f)
    # Required arguments
    req_parser = argparse.ArgumentParser(add_help=False)
    req_group = req_parser.add_argument_group(title="required arguments")
    req_group.add_argument("-d", "--data-dir",
                           type=str,
                           required=True,
                           help="Path to dataset root directory")
    req_group.add_argument("-l", "--log-dir",
                            type=str,
                            dest="log_dir",
                            required=True,
                            metavar="LOGDIR",
                            help="Logdirectory for the session.")
    #Optional arguments
    opt_parser = argparse.ArgumentParser(add_help=False)
    opt_parser.add_argument(
        "-b", "--batch_size",
        type=int,
        dest="batch_size", required=False,
        default=default["hyperparameters"]["batch_size"],
        help="Mini-batch size for stochastic gradient descent algorithm.")
    opt_parser.add_argument(
        "-e", "--epochs",
        type=int,
        dest="epochs", required=False,
        default=default["config"]["epochs"],
        help="How many epochs to do training.")
    opt_parser.add_argument(
        "-lr", "--learning_rate",
        type=float,
        dest="learning_rate", required=False,
        default=default["hyperparameters"]["learning_rate"],
        metavar="LEARNING_RATE",
        help="Initial learning rate.")
    opt_parser.add_argument(
        "--learning_rate_decay",
        type=float,
        dest="lr_decay", required=False,
        default=default["hyperparameters"]["learning_rate_decay"],
        metavar="DECAY",
        help="Learning rate decay factor.")
    opt_parser.add_argument(
        "-s", "--input-size",
        type=int, nargs=3,
        dest="size", required=False,
        default=[default["network"]["input"]["height"],
                 default["network"]["input"]["width"]],
        help="Network input size <height width>.")


    # Create parser hierarchy
    # Top parser
    top_parser = argparse.ArgumentParser(
        usage="%s {cityscapes,freiburg} [-h/--help]"
        % sys.argv[0])

    # Dataset specific parsers inherits required arguments.
    data_parsers = top_parser.add_subparsers(parser_class=HelpfullParser)
    # Cityscapes dataset
    cityscapes = data_parsers.add_parser(
        "cityscapes",
        usage="%s {cityscapes,freiburg} -d DATA_DIR -l LOG_DIR [options]"
        % sys.argv[0],
        parents=[req_parser,opt_parser],
        conflict_handler="resolve",
        help="The Cityscapes dataset.")
    cityscapes.set_defaults(dataset="cityscapes")
    cityscapes.add_argument("-c", "--use-coarse",
                            action="store_true",
                            required=False,
                            dest="coarse")
    # Freiburg forrest dataset
    freiburg = data_parsers.add_parser(
        "freiburg",
        usage="%s {cityscapes,freiburg} -d DATA_DIR -l LOG_DIR [options]"
        % sys.argv[0],
        parents=[req_parser,opt_parser],
        conflict_handler="resolve",
        help="The Freiburg Forest dataset.")
    freiburg.set_defaults(dataset="freiburg")
    freiburg.add_argument("-m", "--modalities",
                          type=str,
                          nargs="+",
                          required=False,
                          default=[],
                          help="Path to Freiburg Forest root directory.")

    args = top_parser.parse_args()
    return vars(args)

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    with open("util/logging.json") as conf:
        conf_dict = json.load(conf)
        logging.config.dictConfig(conf_dict)
        del conf_dict
    args = parse_arguments()
    # Print list of provided arguments
    logger.info(
        "Runnig with following parameters:\n%s" %
        "\n".join(["%-16s : %-s" % (key, value)
                   for key,value in list(args.items())]))
    exit_code = main(args)
    sys.exit(exit_code)
