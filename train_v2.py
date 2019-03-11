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
import util.colormap
import tensortools as tt

def main(args):
    # Handle dataset specific paths and number of classes and paths to
    # training and validation set.
    train_paths = []
    val_paths   = []
    classes = 0
    if args["dataset"] == "cityscapes":
        classes = 19
        train_paths.append(os.path.join(args["data_dir"], "train"))
        if args["coarse"]:
            train_paths.append(os.path.join(args["data_dir"], "train_extra"))
        val_paths.append(os.path.join(args["data_dir"], "val"))

    elif args["dataset"] == "freiburg":
        classes = 6
        train_paths.append(os.path.join(args["data_dir"], "train"))
        val_path = os.path.join(args["data_dir"], "val")
        if os.path.exists(val_path):
            train_paths.append(val_path)

    with tf.device("/device:CPU:0"):
        with tf.name_scope("Datasets"):
            # Setup input pipelines
            input_stage = tt.input.InputStage(args["batch_size"], args["size"])

            # Add datasets
            train_batches = input_stage.add_dataset("train", train_paths,
                                                    epochs=1, augment=True)

            val_batches   = input_stage.add_dataset("val", val_paths, epochs=1)

            # Get iterator outputs
            train_image, train_label, train_mask = input_stage.get_output("train")
            val_image, val_label, val_mask = input_stage.get_output("val")

        # Create step variables
        with tf.variable_scope("StepCounters"):
            global_step = tf.Variable(0, dtype=tf.int64,
                                      trainable=False, name="GlobalStep")
            epoch_step = tf.Variable(0, trainable=False, name="EpochStep")
            epoch_step_inc = tf.assign_add(epoch_step, 1, name="EpochStepInc")

    weight_regularization = None
    if args["l2_reg"] > 0.0:
        weight_regularization = tf.keras.regularizers.l2(args["l2_reg"])
    # Build training and validation network and get prediction output
    train_net = models.ENet(
        classes,
        weight_regularization=weight_regularization
    )
    val_net = models.ENet(classes)
    with tf.device("/device:GPU:0"): #FIXME
        train_logits = train_net(train_image, training=True)
        train_pred = tf.math.argmax(train_logits, axis=-1,
                                    name="TrainPredictions")

    with tf.device("/device:GPU:1"): #FIXME
        val_logits = val_net(val_image, training=False)
        val_pred = tf.math.argmax(val_logits, axis=-1, name="ValPredictions")

    # Build cost function
    with tf.name_scope("Cost"):
        with tf.device("/device:GPU:0"): # FIXME
            # Establish loss function
            with tf.control_dependencies(train_net.updates):
                loss = tt.losses.masked_softmax_cross_entropy(
                    train_label,
                    train_logits,
                    train_mask, classes,
                    #weight=1.02, NOTE NOTE NOTE NOTE NOTE
                    scope="XEntropy")

            cost = loss
            # Add regularization to cost function
            if len(train_net.losses) > 0:
                regularization_loss = tf.math.add_n(train_net.losses, name="Regularization")
                cost += tf.cast(regularization_loss, dtype=tf.float64)

            # Setup learning rate
            learning_rate = args["learning_rate"]
            if args["lr_decay"] > 0.0:
                learning_rate = tf.train.inverse_time_decay(
                    learning_rate, global_step,
                    decay_steps=train_batches, decay_rate=args["lr_decay"])

            # Create optimization procedure
            optimizer = tf.train.AdamOptimizer(learning_rate)

            # Make sure to update the metrics when evaluating loss
            train_op  = optimizer.minimize(cost, global_step=global_step,
                                           name="TrainOp")

    with tf.name_scope("Summary"):
        # Create colormap for image summaries
        colormap = tf.constant(util.colormap.colormap, dtype=tf.uint8,
                               name="Colormap")
        # Create metric evaluation and summaries
        with tf.device("/device:GPU:0"):
            with tf.name_scope("TrainMetrics"):
                train_metrics = tt.metrics.Eval(train_pred, train_label,
                                                classes, train_mask)
                metric_update_op = train_metrics.get_update_op()
                metric_summaries = train_metrics.get_summaries()
                batch_metric_summaries = train_metrics.get_batch_summaries()

            train_summary_iter = tf.summary.merge(
                [
                    batch_metric_summaries["Global"],
                    tf.summary.scalar("CrossEntropyLoss", loss,
                                      family="Losses"),
                    tf.summary.scalar("TotalCost", cost,
                                      family="Losses"),
                    tf.summary.scalar("LearningRate", learning_rate,
                                      family="Losses")
                ], name="IterationSummaries"
               )
            with tf.control_dependencies([metric_update_op]):
                train_summary_epoch = tf.summary.merge(
                    [
                        metric_summaries["Global"],
                        metric_summaries["Class"],
                        metric_summaries["ConfusionMat"],
                    ], name="EpochSummaries"
                   )

        # Create metric evaluation and summaries
        with tf.device("/device:GPU:1"):
            with tf.name_scope("ValidationMetrics"):
                val_metrics = tt.metrics.Eval(val_pred, val_label,
                                              classes, val_mask)
                val_metric_update_op = val_metrics.get_update_op()
                val_metric_summaries = val_metrics.get_summaries()
                val_batch_metric_summaries = val_metrics.get_batch_summaries()

            with tf.control_dependencies([val_metric_update_op]):
                val_summary_epoch = tf.summary.merge(
                    [
                        val_metric_summaries["Global"],
                        val_metric_summaries["Class"],
                        val_metric_summaries["ConfusionMat"],
                        tf.summary.image("Input", val_image),
                        tf.summary.image("Label", tf.gather(
                            colormap, tf.cast(val_label, tf.int32))),
                        tf.summary.image("Predictions", tf.gather(
                            colormap, tf.cast(val_pred, tf.int32)))
                    ], name="EpochSummaries"
                   )

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=sess_config) as sess:
        # Initialize/restore model variables
        logger.debug("Initializing model...")
        sess.run(tf.global_variables_initializer())
        # Prepare fetches
        fetches = {
            "train" : {
                "iteration" : {
                    "step"     : global_step,
                    "summary"  : train_summary_iter,
                    "train_op" : train_op,
                    "update"   : metric_update_op
                },
                "epoch"     : {
                    "step"     : epoch_step,
                    "summary"  : train_summary_epoch
                }
            },
            "val"   : {
                "iteration" : {
                    "update"   : val_metric_update_op
                },
                "epoch"     : {
                    "step"     : epoch_step,
                    "summary"  : val_summary_epoch
                }
            }
        }

        # Create summary writer objects
        summary_writer = tf.summary.FileWriter(args["log_dir"],
                                               graph=sess.graph)
        # Create checkpoint object
        with tf.name_scope("Checkpoint"):
            checkpoint = tf.train.Checkpoint(model=train_net,
                                             epoch=epoch_step,
                                             step=global_step)
            checkpoint_name = os.path.join(args["log_dir"], "model")
            if not os.path.exists(args["log_dir"]):
                os.makedirs(args["log_dir"])

            if args["checkpoint"] is not None:
                # CMDline checkpoint given
                ckpt = args["checkpoint"]
                if os.path.isdir(ckpt):
                    ckpt = tf.train.latest_checkpoint(ckpt)
                if ckpt is None:
                    logger.error("Checkpoint path \"%s\" is invalid.")
                    return 1
                logger.info("Resuming from checkpoint \"%s\"" % ckpt)
                status = checkpoint.restore(ckpt)
                status.assert_existing_objects_matched()
                status.initialize_or_restore(sess)

            elif tf.train.latest_checkpoint(args["log_dir"]) != None:
                # Try to restore from checkpoint in logdir
                ckpt = tf.train.latest_checkpoint(args["log_dir"])
                logger.info("Resuming from checkpoint \"%s\"" % ckpt)
                status = checkpoint.restore(ckpt)
                status.assert_existing_objects_matched()
                status.initialize_or_restore(sess)

            with tf.name_scope("UpdateValidationWeights"):
                update_val_op = []
                for i in range(len(val_net.layers)):
                    for j in range(len(val_net.layers[i].variables)):
                        update_val_op.append(tf.assign(val_net.layers[i].variables[j],
                                                       train_net.layers[i].variables[j]))
                update_val_op = tf.group(update_val_op)
        # END scope Checkpoint

        #run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        logger.info("Starting training loop...")
        results = {}
        for epoch in range(1,args["epochs"]+1):
            # Create iterator counter to track progress
            _iter = range(0,train_batches)
            if show_progress:
                _iter = tqdm.tqdm(_iter, desc="train[%3d/%3d]"
                                  % (epoch, args["epochs"]))
            # Initialize input stage
            input_stage.init_iterator("train", sess)
            input_stage.init_iterator("val", sess)
            # Initialize or update validation network
            sess.run(update_val_op)
            # Reset for another round
            train_metrics.reset_metrics(sess)
            val_metrics.reset_metrics(sess)
            # Prepare initial fetches
            _fetches = {
                "train" : {"iteration" : fetches["train"]["iteration"]},
                "val"   : {"iteration" : fetches["val"]["iteration"]}
            }

            for i in _iter:
                try:
                    # Dynamically update fetches
                    if i == train_batches-1:
                        _fetches["train"]["epoch"] = fetches["train"]["epoch"]
                    if i == val_batches-1:
                        _fetches["val"]["epoch"] = fetches["val"]["epoch"]
                    elif i == val_batches:
                        summary_writer.add_summary(
                            results["val"]["epoch"]["summary"],
                            results["val"]["epoch"]["step"])
                        _fetches.pop("val")
                    # Execute fetches
                    results = sess.run(_fetches
                                        #,options=run_options,
                                        #run_metadata=run_metadata
                    )
                except tf.errors.OutOfRangeError:
                    pass
                # Update summaries
                summary_writer.add_summary(
                    results["train"]["iteration"]["summary"],
                    results["train"]["iteration"]["step"])
                #summary_writer.add_run_metadata(run_metadata, "step=%d" % i)
            summary_writer.add_summary(results["train"]["epoch"]["summary"],
                                       results["train"]["epoch"]["step"])
            summary_writer.flush()
            # Save checkpoint
            checkpoint.save(checkpoint_name, sess)
            # Update epoch counter
            _epoch = sess.run(epoch_step_inc)

        ### FINAL VALIDATION ###
        _fetches = {
            "val" : {"iteration" : dict(fetches["val"]["iteration"])}
                }
        _iter = range(0, val_batches)
        if show_progress:
            _iter = tqdm.tqdm(_iter, desc="val[%3d/%3d]" % (args["epochs"],
                                                            args["epochs"]))
        # Re initialize network
        input_stage.init_iterator("val", sess)
        sess.run(update_val_op)
        for i in _iter:
            try:
                if i >= val_batches-1:
                    _fetches["val"]["epoch"] = fetches["val"]["epoch"]
                results = sess.run(_fetches)
            except tf.errors.OutOfRangeError:
                pass
        # Add final validation summary update
        summary_writer.add_summary(results["val"]["epoch"]["summary"],
                                   results["val"]["epoch"]["step"])
        # Close summary file
        summary_writer.close()
        logger.info("Training successfully finished %d epochs" % args["epochs"])
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
        "-l2", "--l2-regularization",
        type=float,
        dest="l2_reg", required=False,
        default=default["hyperparameters"]["L2_weight"],
        metavar="L2_REGULARIZATION",
        help="L2 Regularization hyperparameter.")
    opt_parser.add_argument(
        "--checkpoint_dir", "-cp",
        type=str,
        dest="checkpoint", required=False,
        metavar="CHECKPOINT",
        help="Path to pretrained checkpoint.")
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
