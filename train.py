# Python standard libraries
import argparse
import json
import logging
import logging.config
import os
import sys

# Non-standard includes
import tensorflow as tf
# Maybe import tqdm
show_progress = False
try:
    import tqdm
    show_progress = True
except ImportError:
    pass

# User includes
import models
import tensortools as tt

def main(args):
    # Handle dataset specific paths and number of classes
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

    # Setup input pipeline
    input_stage = tt.input.InputStage(args["batch_size"], args["size"])
    # Add training dataset
    num_batches = input_stage.add_dataset("train",
                                          data_paths,
                                          augment=True)
    # Get iterator output
    image, label, mask = input_stage.get_output()

    # Setup network and build Network graph
    net = models.ENet(classes)
    logits = net(image, True)
    pred = tf.math.argmax(logits, axis=-1, name="Predictions")

    # Create step variables
    with tf.variable_scope("StepCounters"):
        global_step = tf.Variable(0, dtype=tf.int64,
                                  trainable=False, name="GlobalStep")
        epoch_step = tf.Variable(0, trainable=False, name="EpochStep")
        epoch_step_inc = tf.assign_add(epoch_step, 1, name="EpochStepInc")

    # Build cost function
    with tf.name_scope("Loss"):
        # Establish loss function
        loss = tt.losses.masked_softmax_cross_entropy(
            label, logits, mask, classes, scope="XEntropy")

        # FIXME: insert parameters here:
        optimizer = tf.train.AdamOptimizer(args["learning_rate"])

        # Make sure to update the metrics when evaluating loss
        train_op  = optimizer.minimize(loss, global_step=global_step,
                                       name="TrainOp")

    # Create metric evaluation and summaries
    train_metrics = tt.metrics.Eval(pred, label, classes, mask)
    with tf.name_scope("Summary"):
        metric_summaries = train_metrics.get_summaries()
        batch_metric_summaries = train_metrics.get_batch_summaries()

        summary_iter = tf.summary.merge(
            [
                batch_metric_summaries["Global"],
                tf.summary.scalar("Loss", loss)
            ], name="IterationSummaries"
        )
        summary_epoch = tf.summary.merge(
            [
                metric_summaries["Global"],
                metric_summaries["Class"],
                metric_summaries["ConfusionMat"],
                #TODO: move image summaries to validation thread
                tf.summary.image("Input", image),
                tf.summary.image("Label", tf.expand_dims(label, axis=-1)
                                          * (255//classes)),
                tf.summary.image(
                    "Predictions",
                    tf.expand_dims(
                        tf.cast(pred, dtype=tf.uint8), axis=-1)
                    * (255//classes))
            ], name="EpochSummaries"
        )
    metric_update_op = train_metrics.get_update_op()

    with tf.Session() as sess:

        # Prepare fetches
        fetches = {}
        fetches["iteration"] = {
            "train"   : train_op,
            "step"    : global_step,
            "summary" : summary_iter,
            "metric"  : metric_update_op
        }
        fetches["epoch"] = {
            "summary" : summary_epoch,
            "step"    : epoch_step_inc
        }

        # Initialize variables
        logger.debug("Initializing variables")
        sess.run(tf.global_variables_initializer())

        # Create checkpoint saver object
        vars_to_store = net.variables + [epoch_step, global_step]
        saver   = tf.train.Saver(var_list=vars_to_store, max_to_keep=50)
        savedir = os.path.join(args["log_dir"], "model")
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        elif tf.train.latest_checkpoint(args["log_dir"]) != None:
            ckpt = tf.train.latest_checkpoint(args["log_dir"])
            logger.info("Resuming from checkpoint \"%s\"" % ckpt)
            saver.restore(sess, ckpt)
        # Create summary writer object
        summary_writer = tf.summary.FileWriter(args["log_dir"],
                                               graph=sess.graph)
       # run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
       # run_metadata = tf.RunMetadata()
        logger.info("Starting training loop...")
        results = {}
        for epoch in range(1,args["epochs"]+1):
            # Create iterator counter to track progress
            _iter = range(0,num_batches)
            _iter = _iter if not show_progress \
                          else tqdm.tqdm(_iter, desc="train[%3d/%3d]"
                                         % (epoch, args["epochs"]))
            # Initialize input stage
            input_stage.init_iterator("train", sess)
            # Reset for another round
            train_metrics.reset_metrics(sess)

            for i in _iter:
                try:
                    _fetches = {"iteration" : fetches["iteration"]} \
                               if i < num_batches-1 else fetches
                    results = sess.run(_fetches
                    #                    options=run_options,
                    #                    run_metadata=run_metadata
                    )
                except tf.errors.OutOfRangeError:
                    pass
                summary_writer.add_summary(results["iteration"]["summary"],
                                           results["iteration"]["step"])
                #summary_writer.add_run_metadata(run_metadata, "step=%d" % i)
            summary_writer.add_summary(results["epoch"]["summary"],
                                       results["epoch"]["step"])
            summary_writer.flush()
            saver.save(sess, savedir, global_step=results["epoch"]["step"])
    return 0


class HelpfullParser(argparse.ArgumentParser):
    # Prints help instead of usage string on error
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
        type=int, nargs=2,
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
    if not "freiburg" in sys.argv and \
       not "cityscapes" in sys.argv:
        top_parser.print_help()
        sys.exit(0)
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
