# Python standard libraries
import argparse
import json
import logging
import logging.config
import os
import pprint
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
import datasets
import tensortools as tt

def main(args):
    # Retrieve dataset specific object
    if args.dataset == "cityscapes":
        dataset = datasets.Cityscapes(coarse=args.coarse)
    elif args.dataset == "freiburg":
        dataset = datasets.Freiburg()
    elif args.dataset == "vistas":
        dataset = datasets.Vistas()
    else:
        raise NotImplementedError("Dataset \"%s\" not supported" % args.dataset)
    # Gather train and validation paths
    train_paths = os.path.join(args.data_dir, "train")
    val_paths   = os.path.join(args.data_dir, "val")
    # Retrieve training parameters
    params  = args.params
    hparams = params["hyperparams"]

    with tf.device("/device:CPU:0"):
        with tf.name_scope("Datasets"):
            # Setup input pipelines
            train_input = tt.input.InputStage(
                input_shape=[params["network"]["input"]["height"],
                             params["network"]["input"]["width"]])
            val_input   = tt.input.InputStage(
                input_shape=[params["network"]["input"]["height"],
                             params["network"]["input"]["width"]])

            # Add datasets
            train_examples = train_input.add_dataset(
                    "train", train_paths,
                    batch_size=params["batch_size"],
                    epochs=1, augment=True)
            val_examples   = val_input.add_dataset(
                    "val", val_paths,
                    batch_size=params["batch_size"],
                    epochs=1)
            # Calculate number of batches
            train_batches = (train_examples-1)//params["batch_size"] + 1
            val_batches   = (val_examples - 1)//params["batch_size"] + 1

            # Get iterator outputs
            _, train_image, train_label, train_mask = train_input.get_output()
            val_image, val_label, val_mask = val_input.get_output()

        # Create step variables
        with tf.variable_scope("StepCounters"):
            # I'll use one local (to this run) and a global step that
            # will be checkpointed in order to run various schedules on
            # the learning rate decay policy.
            global_step = tf.Variable(0, dtype=tf.int64,
                                      trainable=False, name="GlobalStep")
            local_step = tf.Variable(0, dtype=tf.int64,
                                      trainable=False, name="LocalStep")
            global_step_op = global_step + local_step
            epoch_step = tf.Variable(0, trainable=False, name="EpochStep")
            epoch_step_inc = tf.assign_add(epoch_step, 1, name="EpochStepInc")

    regularization = {}
    if hparams["weight_reg"]["L2"] > 0.0 \
        or hparams["weight_reg"]["L1"] > 0.0:
        regularization = {
            "weight_regularization"  : tf.keras.regularizers.l1_l2(
                                           l1=hparams["weight_reg"]["L1"],
                                           l2=hparams["weight_reg"]["L2"]),
            "regularization_scaling" : hparams["weight_reg"]["glorot_scaling"]
        }
    # Build training and validation network and get prediction output
    train_net = models.ENet(
        dataset.num_classes,
        **regularization
    )
    val_net = models.ENet(dataset.num_classes)
    with tf.device("/device:GPU:0"):
        train_logits = train_net(train_image, training=True)
        train_pred   = tf.math.argmax(train_logits, axis=-1,
                                      name="TrainPredictions")

    with tf.device("/device:GPU:1"):
        val_logits = val_net(val_image, training=False)
        val_pred   = tf.math.argmax(val_logits, axis=-1,
                                    name="ValidationPredictions")

    # Build cost function
    with tf.name_scope("Cost"):
        with tf.device("/device:GPU:0"):
            # Establish loss function
            if hparams["softmax"]["multiscale"]:
                loss, loss_weights = \
                        tt.losses.multiscale_masked_softmax_cross_entropy(
                    train_label,
                    train_net.endpoint_outputs[0],
                    train_mask, dataset.num_classes,
                    weight=hparams["softmax"]["loginverse_scaling"],
                    label_smoothing=hparams["softmax"]["label_smoothing"],
                    scope="XEntropy")
                # NOTE: this will make @loss_weights checkpointed
                train_net.loss_scale_weights = loss_weights
            else:
                loss = tt.losses.masked_softmax_cross_entropy(
                    train_label,
                    train_logits,
                    train_mask, dataset.num_classes,
                    weight=hparams["softmax"]["loginverse_scaling"],
                    label_smoothing=hparams["softmax"]["label_smoothing"],
                    scope="XEntropy")
            cost = loss
            # Add regularization to cost function
            if len(train_net.losses) > 0:
                regularization_loss = tf.math.add_n(train_net.losses, name="Regularization")
                cost += tf.cast(regularization_loss, dtype=tf.float64)

            # Setup learning rate
            learning_rate = hparams["learning_rate"]
            if hparams["learning_rate_decay"] > 0.0:
                # Inverse time learning_rate if lr_decay specified
                learning_rate = tf.train.inverse_time_decay(
                    learning_rate, local_step,
                    decay_steps=train_batches,
                    decay_rate=hparams["learning_rate_decay"])

            # Create optimization procedure
            optimizer = tf.train.AdamOptimizer(learning_rate, **hparams["optimizer"]["kwargs"])

            # Create training op
            train_op  = optimizer.minimize(cost, global_step=local_step,
                                           name="TrainOp")
            # NOTE: Make sure to update batchnorm params and metrics for
            # each training iteration.

    # Create summary operations for training and validation network
    with tf.name_scope("Summary"):
        # Create colormap for image summaries
        colormap = tf.constant(dataset.colormap, dtype=tf.uint8,
                               name="Colormap")
        # Create metric evaluation and summaries
        with tf.device("/device:GPU:0"):
            with tf.name_scope("TrainMetrics"):
                train_metrics = tt.metrics.Metrics(train_pred, train_label,
                                                   dataset.num_classes, train_mask)
                metric_update_op = train_metrics.get_update_op()
                metric_summaries = train_metrics.get_summaries()

            train_summary_iter = tf.summary.merge(
                [
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
                        metric_summaries["Metrics"],
                        metric_summaries["ConfusionMat"],
                    ], name="EpochSummaries"
                   )

        # Create metric evaluation and summaries
        with tf.device("/device:GPU:1"):
            with tf.name_scope("ValidationMetrics"):
                val_metrics = tt.metrics.Metrics(val_pred, val_label,
                                              dataset.num_classes, val_mask)
                val_metric_update_op = val_metrics.get_update_op()
                val_metric_summaries = val_metrics.get_summaries()

                with tf.control_dependencies([val_metric_update_op]):
                    val_summary_epoch = tf.summary.merge(
                        [
                            val_metric_summaries["Metrics"],
                            val_metric_summaries["ClassMetrics"],
                            val_metric_summaries["ConfusionMat"],
                            tf.summary.image("Input", val_image),
                            tf.summary.image("Label", tf.gather(
                                colormap, tf.cast(val_label + 255*(1-val_mask),
                                                  tf.int32))),
                            tf.summary.image("Predictions", tf.gather(
                                colormap, tf.cast(val_pred, tf.int32)))
                        ], name="EpochSummaries"
                    )
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        # Dump parameter configuration (args)
        with open(os.path.join(args.log_dir, "config.json"), "w+") as f:
            json.dump(params, f, indent=4, sort_keys=True)

    # Create session with soft device placement
    #     - some ops neet to run on the CPU
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=sess_config) as sess:
        # Initialize/restore model variables
        logger.debug("Initializing model...")
        sess.run(tf.global_variables_initializer())
        # Create summary writer objects
        summary_writer = tf.summary.FileWriter(args.log_dir,
                                               graph=sess.graph)

        # Create checkpoint object
        with tf.name_scope("Checkpoint"):
            checkpoint = tf.train.Checkpoint(model=train_net,
                                             epoch=epoch_step,
                                             step=global_step,
                                             optimizer=optimizer)
            checkpoint_name = os.path.join(args.log_dir, "model")

            if args.checkpoint is not None:
                # CMDline checkpoint given
                ckpt = args.checkpoint
                if os.path.isdir(ckpt):
                    ckpt = tf.train.latest_checkpoint(ckpt)
                if ckpt is None:
                    logger.error("Checkpoint path \"%s\" is invalid.")
                    return 1
                logger.info("Resuming from checkpoint \"%s\"" % ckpt)
                status = checkpoint.restore(ckpt)
                if tf.__version__ < "1.14.0":
                    status.assert_existing_objects_matched()
                else:
                    status.expect_partial()
                status.initialize_or_restore(sess)

            elif tf.train.latest_checkpoint(args.log_dir) != None:
                # Try to restore from checkpoint in logdir
                ckpt = tf.train.latest_checkpoint(args.log_dir)
                logger.info("Resuming from checkpoint \"%s\"" % ckpt)
                status = checkpoint.restore(ckpt)
                if tf.__version__ < "1.14.0":
                    status.assert_existing_objects_matched()
                else:
                    status.expect_partial()
                status.initialize_or_restore(sess)

            with tf.name_scope("UpdateValidationWeights"):
                update_val_op = []
                for i in range(len(val_net.layers)):
                    for j in range(len(val_net.layers[i].variables)):
                        update_val_op.append(tf.assign(val_net.layers[i].variables[j],
                                                       train_net.layers[i].variables[j]))
                update_val_op = tf.group(update_val_op)
        # END scope Checkpoint

        # Prepare fetches
        fetches = {
            "train" : {
                "iteration" : {
                    "step"     : global_step_op,
                    "summary"  : train_summary_iter,
                    "train_op" : train_op,
                    "update"   : metric_update_op,
                    "updates"  : train_net.updates
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
        #run_options  = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        #run_metadata = tf.RunMetadata()
        logger.info("Starting training loop...")
        results = {}
        for epoch in range(1,params["epochs"]+1):
            # Create iterator counter to track progress
            _iter = range(0,train_batches)
            if show_progress:
                _iter = tqdm.tqdm(_iter, desc="train[%3d/%3d]"
                                  % (epoch, params["epochs"]),
                                  ascii=True,
                                  dynamic_ncols=True)
            # Initialize input stage
            train_input.init_iterator("train", sess)
            val_input.init_iterator("val", sess)
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

            # Update epoch counter
            _epoch = sess.run(epoch_step_inc)

            # Update epoch summaries
            summary_writer.add_summary(results["train"]["epoch"]["summary"],
                                       results["train"]["epoch"]["step"])
            summary_writer.flush()
            # Save checkpoint
            checkpoint.save(checkpoint_name, sess)

        ### FINAL VALIDATION ###
        _fetches = {
            "val" : {"iteration" : fetches["val"]["iteration"]}
        }
        _iter = range(0, val_batches)
        if show_progress:
            _iter = tqdm.tqdm(_iter, desc="val[%3d/%3d]" % (params["epochs"],
                                                            params["epochs"]))
        # Re initialize network
        val_input.init_iterator("val", sess)
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
        logger.info("Training successfully finished %d epochs" % params["epochs"])
    return 0


class HelpfullParser(argparse.ArgumentParser):
    # Prints help instead of usage string on error
    def error(self, message):
        self.print_help()
        self.exit(2, "error: %s\n" % message)

def parse_arguments():
    """
    Handles parseing of commandline arguments

    :returns: The parsed commandline options
    :rtype:   dict
    """
    # Required arguments
    req_parser = argparse.ArgumentParser(add_help=False)
    req_group = req_parser.add_argument_group(title="Required arguments")
    req_group.add_argument("-d", "--data-dir",
                           type=str,
                           dest="data_dir",
                           required=True,
                           help="Path to dataset root directory")
    req_group.add_argument("-l", "--log-dir",
                            type=str,
                            dest="log_dir",
                            required=True,
                            metavar="LOGDIR",
                            help="Logdirectory for the session.")
    req_group.add_argument("-p", "--parameters",
                            type=str,
                            dest="params",
                            default="conf/default_params.json",
                            metavar="PARAMS",
                            help="Path to parameter configuration file, "
                                 "see conf subdirectory.")

    #Optional arguments
    opt_parser = argparse.ArgumentParser(add_help=False)
    opt_parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        dest="checkpoint", required=False,
        metavar="CHECKPOINT",
        help="Path to pretrained checkpoint directory or model."
    )

    # Create parser hierarchy
    # Top parser
    top_parser = argparse.ArgumentParser(
        usage="%s {cityscapes,freiburg,vistas} [-h/--help]"
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
    cityscapes.add_argument("--use-coarse",
                            action="store_true",
                            required=False,
                            dest="coarse")
    # Mapillary Vistas dataset
    vistas = data_parsers.add_parser(
        "vistas",
        usage="%s {cityscapes,freiburg,vistas} -d DATA_DIR -l LOG_DIR [options]"
        % sys.argv[0],
        parents=[req_parser,opt_parser],
        conflict_handler="resolve",
        help="The Mapillary Vistas dataset.")
    vistas.set_defaults(dataset="vistas")

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
       not "cityscapes" in sys.argv and \
       not "vistas" in sys.argv:
        top_parser.print_help()
        sys.exit(0)
    args = top_parser.parse_args()

    return args

if __name__ == "__main__":
    # Get and configure logger
    logger = logging.getLogger(__name__)
    with open("util/logging.json") as conf:
        conf_dict = json.load(conf)
        logging.config.dictConfig(conf_dict)
        del conf_dict
    args = parse_arguments()
    # Load parameters
    parameters = None
    with open(args.params, "r") as f:
        parameters = json.load(f)
    # Overwrite with parameter dict
    args.params = parameters
    # Print list of provided arguments

    logger.info(
        "Runnig with following parameters:\n%s" %
        json.dumps(parameters, sort_keys=True, indent=4))
    exit_code = main(args)
    sys.exit(exit_code)
