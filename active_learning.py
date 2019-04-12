# Python standard libraries
import argparse
import glob
import json
import logging
import logging.config
import os
import sys

# Non-standard includes
import numpy as np
import tensorflow as tf
# Maybe import tqdm
show_progress = False
try:
    import tqdm
    show_progress = True
except ImportError:
    pass

try:
    import tkinter
    tkinter.Tk().withdraw()
except ImportError:
    if args.unlabelled == None:
        pass
    else:
        raise ImportError("Could not import tkinter, make sukre Tk "
                          "dependencies are installed")

# User includes
import models
import datasets
import tensortools as tt

# Lowest representable float32
EPSILON = np.finfo(np.float32).tiny

def main(args, logger):
    # Retrieve training parameters for convenience
    params   = args.params               # All parameters
    hparams  = params["hyperparams"]     # Hyperparamters
    alparams = params["active_learning"] # Active learning parameters
    state = None # State dict
    # Define state and config filenames
    state_filename  = os.path.join(args.log_dir, "state.json")
    config_filename = os.path.join(args.log_dir, "config.json")
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        # Dump parameter config
        with open(config_filename, "w+") as f:
            json.dump(params, f, indent=4)

    # Retrieve dataset specific object
    if args.dataset == "cityscapes":
        dataset = datasets.Cityscapes(coarse=args.coarse)
        test_examples_glob = os.path.join(args.data_dir, "val", "*.tfrecord")
    elif args.dataset == "freiburg":
        dataset = datasets.Freiburg()
        test_examples_glob = os.path.join(args.data_dir, "test", "*.tfrecord")
    elif args.dataset == "vistas":
        dataset = datasets.Vistas()
        test_examples_glob = os.path.join(args.data_dir, "val", "*.tfrecord")
    else:
        raise NotImplementedError("Dataset \"%s\" not supported" % args.dataset)

    # Prepare dataset example file paths.
    train_examples_glob = os.path.join(args.data_dir, "train", "*.tfrecord")

    if not os.path.exists(state_filename):
        # Initialize state
        # Resolve example filenames
        train_val_examples = np.sort(np.array(glob.glob(train_examples_glob)))
        # Pick examples from training set to use for validation
        val_examples   = train_val_examples[:alparams["num_validation"]]
        # Use the rest as training examples
        train_examples = train_val_examples[alparams["num_validation"]:]

        # Use annotated test set, NOTE: cityscapes validation set
        test_examples  = np.array(glob.glob(test_examples_glob))

        # Draw random train examples and mark as annotated
        train_indices  = np.arange(len(train_examples), dtype=np.int32)
        np.random.shuffle(train_indices)

        initially_labelled = alparams["num_initially_labelled"]
        if initially_labelled < 0:
            # Use rest of labelled examples
            initially_labelled = len(train_examples)

        # Possibly add actually unlabelled examples
        no_label_indices = np.empty(0, dtype=str)
        if args.unlabelled is not None:
            no_label_glob     = os.path.join(args.unlabelled, "*.tfrecord")
            no_label_examples = glob.glob(no_label_glob)
            no_label_indices  = np.arange(
                len(train_indices), len(train_indices)+len(no_label_examples)
            )
            train_examples = np.concatenate(train_examples,
                                            no_label_examples)
            train_indices = np.concatenate((train_indices, no_label_indices))

        labelled = train_indices[:initially_labelled]
        unlabelled = train_indices[initially_labelled:]
        del train_indices

        # Setup initial state
        state = {
            "checkpoint" : None, # Keep track of latest checkpoint.
            "iteration"  : 0,
            "dataset" : {
                "train" : {
                    "filenames"  : list(train_examples),
                    "labelled"   : labelled.tolist(),
                    "unlabelled" : unlabelled.tolist(),
                    "no_label"   : no_label_indices.tolist()
                },
                "val"   : {
                    "filenames" : list(val_examples)
                },
                "test"  : {
                    "filenames" : list(test_examples)
                }
            }
        }
        with open(state_filename, "w+") as f:
            json.dump(state, f, indent=2)

    else:
        # Load state
        with open(state_filename, "r") as f:
            state = json.load(f)
        # Extract filename properties
        train_examples   = np.array(state["dataset"]["train"]["filenames"])
        val_examples     = np.array(state["dataset"]["val"]["filenames"])
        test_examples    = np.array(state["dataset"]["test"]["filenames"])
        labelled         = np.array(state["dataset"]["train"]["labelled"])
        unlabelled       = np.array(state["dataset"]["train"]["unlabelled"])
        no_label_indices = np.array(state["dataset"]["train"]["no_label"])

    # TODO dump this into log dir, and dump modified version in log subdirs
    train_input_labelled = np.full_like(train_examples, False, dtype=bool)
    train_input_labelled[labelled] = True
    train_input_indices = np.arange(len(train_examples))

    with tf.device("/device:CPU:0"):
        with tf.name_scope("Datasets"):
            # Create input placeholders
            train_input = tt.input.NumpyCapsule()
            train_input.filenames = train_examples
            train_input.labelled = train_input_labelled
            train_input.indices   = train_input_indices

            val_input = tt.input.NumpyCapsule()
            val_input.filenames = val_examples
            test_input = tt.input.NumpyCapsule()
            test_input.filenames = test_examples

            # Setup input pipelines
            train_input_stage = tt.input.InputStage(
                input_shape=[params["network"]["input"]["height"],
                             params["network"]["input"]["width"]])
            # Validation AND Test input stage
            val_input_stage  = tt.input.InputStage(
                input_shape=[params["network"]["input"]["height"],
                             params["network"]["input"]["width"]])

            # Add datasets
            train_input_stage.add_dataset_from_placeholders(
                "train", train_input.filenames,
                train_input.labelled, train_input.indices,
                batch_size=params["batch_size"],
                augment=True)
            # Validation set
            val_input_stage.add_dataset_from_placeholders(
                "val", val_input.filenames,
                batch_size=params["batch_size"])
            # Test set
            val_input_stage.add_dataset_from_placeholders(
                "test", test_input.filenames,
                batch_size=params["batch_size"])
            # Calculate number of batches in each iterator
            val_batches   = (len(val_examples) - 1)//params["batch_size"] + 1
            test_batches  = (len(test_examples) - 1)//params["batch_size"] + 1

            # Get iterator outputs
            train_image, train_label, train_mask, train_labelled, train_index \
                                               = train_input_stage.get_output()
            val_image, val_label, val_mask = val_input_stage.get_output()

        # Create step variables
        with tf.variable_scope("StepCounters"):
            global_step = tf.Variable(0, dtype=tf.int64,
                                      trainable=False, name="GlobalStep")
            local_step  = tf.Variable(0, dtype=tf.int64,
                                      trainable=False, name="LocalStep")
            global_step_op = tf.assign_add(global_step, local_step)
            epoch_step  = tf.Variable(0, trainable=False, name="EpochStep")
            epoch_step_inc = tf.assign_add(epoch_step, 1)

    # Build training- and validation network
    regularization = {}
    if hparams["weight_reg"]["L2"] > 0.0 \
       or hparams["weight_reg"]["L1"] > 0.0:
        regularization = {
            "weight_regularization" : tf.keras.regularizers.l1_l2(
                                          l1=hparams["weight_reg"]["L1"],
                                          l2=hparams["weight_reg"]["L2"]),
            "regularization_scaling" : hparams["weight_reg"]["glorot_scaling"]
        }

    # Initialize networks
    train_net = models.ENet(
        dataset.num_classes,
        **regularization
    )
    val_net = models.ENet(dataset.num_classes)

    with tf.device("/device:GPU:0"):
        # Build graph for training
        train_logits  = train_net(train_image, training=True)
        # Compute predictions: use @train_pred for metrics and
        # @pseudo_label for pseudo_annotation process.
        train_pred    = tf.math.argmax(train_logits, axis=-1,
                                       name="TrainPredictions")

        with tf.name_scope("PseudoAnnotation"):
            # Build ops one more time without dropout.
            pseudo_logits = train_net(train_image, training=False)
            # Just make sure not to propagate gradients a second time.
            pseudo_logits = tf.stop_gradient(pseudo_logits)
            pseudo_label  = tf.math.argmax(pseudo_logits, axis=-1,
                                           name="TrainPredictions")
            pseudo_label = tf.cast(pseudo_label, tf.uint8)

            # Configure on-line high confidence pseudo labeling.
            pseudo_prob   = tf.nn.softmax(pseudo_logits, axis=-1, name="TrainProb")
            if alparams["measure"] == "entropy":
                # Reduce entropy over last dimension.
                # Compute prediction entropy
                entropy = - pseudo_prob * tf.math.log(pseudo_prob+EPSILON)
                entropy = tf.math.reduce_sum(entropy, axis=-1)
                # Convert logarithm base to units of number of classes
                # NOTE this will make the metric independent of number of
                #      classes as well the range in [0,1]
                log_base = tf.math.log(np.float32(dataset.num_classes))
                entropy = entropy / log_base
                # Convert entropy to confidence
                pseudo_confidence = 1.0 - entropy
            elif alparams["measure"] == "margin":
                # Difference between the two largest entries in last dimension.
                values, indices = tf.math.top_k(pseudo_prob, k=2)
                pseudo_confidence = values[:,:,:,0] - values[:,:,:,1]
            elif alparams["measure"] == "confidence":
                # Reduce max over last dimension.
                pseudo_confidence = tf.math.reduce_max(pseudo_prob, axis=-1)
            else:
                raise NotImplementedError("Uncertainty function not implemented.")
            pseudo_mean_confidence = tf.reduce_mean(
                tf.cast(pseudo_confidence, tf.float64),
                axis=(1,2))
            # Pseudo annotate high-confidence unlabeled example pixels
            pseudo_mask = tf.where(tf.math.less(pseudo_confidence, alparams["threshold"]),
                                   tf.zeros_like(pseudo_label,
                                                 dtype=train_label.dtype),
                                   tf.ones_like(pseudo_label,
                                                dtype=train_label.dtype))
            # Pseudo annotation logic (think of it as @tf.cond maped 
            # over batch dimension)
            train_label = tf.where(train_labelled, train_label,
                                   pseudo_label, name="MaybeGenLabel")
            train_mask  = tf.where(train_labelled, train_mask,
                                   pseudo_mask, name="MaybeGenMask")

    with tf.device("/device:GPU:1"):
        # Build validation network.
        val_logits = val_net(val_image, training=False)
        val_pred   = tf.math.argmax(val_logits, axis=-1,
                                    name="ValidationPredictions")

    # Build cost function
    with tf.name_scope("Cost"):
        with tf.device("/device:GPU:0"):
            # Establish loss function
            if hparams["softmax"]["multiscale"]:
                #FIXME: can't use train_label / train_mask here
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
        # END tf.device("/device:GPU:0")
    # END tf.name_scope("Cost")

    # Create summary operations for training and validation network
    with tf.name_scope("Summary"):
        # Create colormap for image summaries
        colormap = tf.constant(dataset.colormap, dtype=tf.uint8,
                               name="Colormap")
        # Create metric evaluation and summaries
        with tf.device("/device:GPU:0"):
            with tf.name_scope("TrainMetrics"):
                # Create metrics object for training network.
                train_metrics = tt.metrics.Metrics(train_pred, train_label,
                                                   dataset.num_classes, train_mask)
                # Get Tensorflow update op.
                metric_update_op = train_metrics.get_update_op()
                # Get Tensorflow summary operations.
                metric_summaries = train_metrics.get_summaries()

            train_summary_iter = tf.summary.merge(
                [
                    # Summaries run at each iteration.
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
                        # Summaries run at epoch boundaries.
                        metric_summaries["Metrics"],
                        metric_summaries["ConfusionMat"],
                    ], name="EpochSummaries"
                   )
        # Create metric evaluation and summaries
        with tf.device("/device:GPU:1"):
            with tf.name_scope("ValidationTestMetrics"):
                # Create metrics object
                val_metrics = tt.metrics.Metrics(val_pred, val_label,
                                                 dataset.num_classes, val_mask)
                # Get update tensorflow ops
                val_metric_update_op = val_metrics.get_update_op()
                # Get metric sumaries
                val_metric_summaries = val_metrics.get_summaries()

                with tf.control_dependencies([val_metric_update_op]):
                    val_metric_summary = tf.summary.merge(
                        [
                            # "Expensive" summaries run at epoch boundaries.
                            val_metric_summaries["Metrics"],
                            val_metric_summaries["ClassMetrics"],
                            val_metric_summaries["ConfusionMat"]
                        ], name="EpochSummaries"
                    )
                    val_image_summary = tf.summary.merge(
                        [
                            tf.summary.image("Input", val_image),
                            tf.summary.image("Label", tf.gather(
                                colormap, tf.cast(val_label + 255*(1-val_mask),
                                                  tf.int32))),
                            tf.summary.image("Predictions", tf.gather(
                                colormap, tf.cast(val_pred, tf.int32)))
                        ]
                    )
                    val_summary_epoch = val_metric_summary
                    test_summary_epoch = tf.summary.merge([
                        val_metric_summary,
                        val_image_summary
                        ])
    # END name_scope("Summary")

    # Create session with soft device placement
    #     - some ops neet to run on the CPU
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        logger.debug("Initializing variables...")
        sess.run(tf.global_variables_initializer())

        # TODO read json file giving current progression and filenames

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
                status.assert_existing_objects_matched()
                status.initialize_or_restore(sess)

            elif state["checkpoint"] != None:
                # Try to restore from checkpoint in logdir
                ckpt = state["checkpoint"]
                logger.info("Resuming from checkpoint \"%s\"" % ckpt)
                status = checkpoint.restore(ckpt)
                status.assert_existing_objects_matched()
                status.initialize_or_restore(sess)

            with tf.name_scope("UpdateValidationWeights"):
                update_val_op = []
                for i in range(len(val_net.layers)):
                    for j in range(len(val_net.layers[i].variables)):
                        update_val_op.append(
                            tf.assign(val_net.layers[i].variables[j],
                                      train_net.layers[i].variables[j]))
                update_val_op = tf.group(update_val_op)

        ckpt_manager = tt.checkpoint_manager.CheckpointManager(checkpoint,
                                                           args.log_dir)
        # END scope Checkpoint
        # Prepare global fetches dict
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
            "val"   : { # Validation and test fetches
                "iteration" : {
                    "update"   : val_metric_update_op
                },
                "epoch"     : {
                    "step"     : epoch_step,
                    "MeanIoU"  : val_metrics.metrics["MeanIoU"],
                    "summary"  : val_summary_epoch,
                    # Also add image summary, however only added to
                    # writer every N epochs.
                    "summary/image" : val_image_summary
                }
            },
            "test" : {
                "iteration" : {"update"  : val_metric_update_op},
                "epoch"     : {"summary" : test_summary_epoch}
            }
        }

        # Train loop (until convergence) -> Pick unlabeled examples -> test_loop
        def train_loop(summary_writer, sample_unlabelled=0):
            """
            Train loop closure.
            Runs training loop untill no improvement is seen in
            @params["epochs"] epochs before returning.
            """
            _initial_grace_period = 50 # How many epoch until counting @no_improvement
            best_ckpt             = state["checkpoint"]
            no_improvement_count  = 0
            best_mean_iou         = 0.0
            log_subdir            = summary_writer.get_logdir()
            run_name              = os.path.basename(log_subdir)
            checkpoint_prefix     = os.path.join(log_subdir, "model")
            num_iter_per_epoch    = np.maximum(train_input.size,
                                              val_input.size)
            while no_improvement_count < params["epochs"] \
                    and _initial_grace_period >= 0:
                _initial_grace_period -= 1
                # Increment in-graph epoch counter.
                epoch = sess.run(epoch_step_inc)

                # Prepare inner loop iterator
                _iter = range(0, num_iter_per_epoch, params["batch_size"])
                if show_progress:
                    _iter = tqdm.tqdm(_iter, desc="%s[%d]" % (run_name, epoch),
                                      dynamic_ncols=True,
                                      ascii=True,
                                      postfix={"NIC" : no_improvement_count})

                # Initialize iterators
                train_input_stage.init_iterator(
                    "train", sess, train_input.feed_dict)
                val_input_stage.init_iterator(
                    "val", sess, val_input.feed_dict)

                # Reset confusion matrices
                train_metrics.reset_metrics(sess)
                val_metrics.reset_metrics(sess)

                # Prepare iteration fetches
                _fetches = {
                    "train" : {"iteration" : fetches["train"]["iteration"]},
                    "val"   : {"iteration" : fetches["val"]["iteration"]}
                }
                # Update validation network weights
                sess.run(update_val_op)

                try:
                    for i in _iter:
                        if train_input.size-params["batch_size"] <= i < train_input.size:
                            # Fetches for last training iteration.
                            _fetches["train"]["epoch"] = fetches["train"]["epoch"]
                        if val_input.size-params["batch_size"] <= i < val_input.size:
                            _fetches["val"]["epoch"] = fetches["val"]["epoch"]

                        # Run fetches
                        results = sess.run(_fetches)

                        if "train" in results.keys():
                            # Add iteration summary
                            summary_writer.add_summary(
                                results["train"]["iteration"]["summary"],
                                results["train"]["iteration"]["step"])

                            # Maybe add epoch summary
                            if "epoch" in results["train"].keys():
                                summary_writer.add_summary(
                                    results["train"]["epoch"]["summary"],
                                    results["train"]["epoch"]["step"]
                                )
                                # Pop fetches to prohibit OutOfRangeError due to
                                # asymmetric train-/val- input size.
                                _fetches.pop("train")
                        if "val" in results.keys() and \
                           "epoch" in results["val"].keys():
                            # Add summaries to event log.
                            summary_writer.add_summary(
                                results["val"]["epoch"]["summary"],
                                results["val"]["epoch"]["step"]
                            )
                            if results["val"]["epoch"]["step"] % 100 == 0:
                                # Only report image summary every 100th epoch.
                                summary_writer.add_summary(
                                    results["val"]["epoch"]["summary/image"],
                                    results["val"]["epoch"]["step"]
                                )
                            # Check if MeanIoU improved and
                            # update counter and best
                            if results["val"]["epoch"]["MeanIoU"] > best_mean_iou:
                                best_mean_iou = results["val"]["epoch"]["MeanIoU"]
                                # Update checkpoint file used for
                                # @tf.train.latest_checkpoint to point at
                                # current best.
                                _ckpt_name = ckpt_manager.commit(
                                    checkpoint_prefix, sess)
                                if _ckpt_name != "":
                                    best_ckpt = _ckpt_name
                                # Reset counter
                                no_improvement_count = 0
                            else:
                                # Result has not improved, increment counter.
                                no_improvement_count += 1
                                if no_improvement_count >= params["epochs"] and \
                                   _initial_grace_period < 0:
                                    _iter.close()
                                    break
                            # Pop fetches to prohibit OutOfRangeError due to
                            # asymmetric train-/val- input size.
                            _fetches.pop("val")
                        # END "maybe add epoch summary"
                except tf.errors.OutOfRangeError:
                    logger.error("Out of range error. Attempting to continue.")
                    pass

                summary_writer.flush()
                ckpt_manager.cache(sess)
            # END while no_improvement_count < params["epochs"]
            return best_ckpt

        def test_loop(summary_writer):
            """
            Test loop closure.
            """
            _step = len(labelled)
            # Initialize validation input stage with test set
            val_input_stage.init_iterator("test", sess, test_input.feed_dict)
            _iter = range(0, test_input.size, params["batch_size"])
            if show_progress:
                _iter = tqdm.tqdm(_iter, desc="test[%d]" % (_step),
                                  ascii=True,
                                  dynamic_ncols=True)
            summary_proto = None
            val_metrics.reset_metrics(sess)
            try:
                for i in _iter:
                    # Accumulate confusion matrix
                    if i < test_input.size - params["batch_size"]:
                        sess.run(fetches["test"]["iteration"]["update"])
                    else:
                        # Run summary operation last iteration
                        _, summary_proto = sess.run([fetches["test"]["iteration"]["update"],
                                                     fetches["test"]["epoch"]["summary"]])
            except tf.errors.OutOfRangeError:
                pass
            # Add summary with number of labelled examples as step.
            # NOTE this only runs on each major iteration.
            summary_writer.add_summary(
                summary_proto, _step
            )

        def rank_confidence():
            # Allocate array to store all confidence scores
            num_examples = len(state["dataset"]["train"]["filenames"])
            confidence = np.zeros(num_examples, dtype=np.float32)
            # Initialize input stage
            train_input_stage.init_iterator("train", sess,
                                            train_input.feed_dict)
            _iter = range(0, train_input.size, params["batch_size"])
            if show_progress:
                _iter = tqdm.tqdm(_iter, desc="ranking[%d]" % len(labelled),
                                  ascii=True,
                                  dynamic_ncols=True)
            try:
                for i in _iter:
                    # Loop over all examples and compute confidence
                    batch_confidence, batch_indices = sess.run(
                        [pseudo_mean_confidence, train_index])
                    # Add to list of confidence
                    confidence[batch_indices] = batch_confidence
            except tf.errors.OutOfRangeError:
                pass

            # Filter out labelled examples
            unlabelled_confidence = confidence[unlabelled]

            selection_size = np.minimum(len(unlabelled),
                                        alparams["selection_size"])
            # Get the lowest confidence indices of unlabelled subset
            example_indices = np.argpartition(unlabelled_confidence,
                                              selection_size)
            example_indices = example_indices[:selection_size]
            # Convert to indices into all filenames list
            example_indices = unlabelled[example_indices]
            return example_indices

        checkpoint_path = state["checkpoint"]
        # Only add graph to first event file
        _graph = sess.graph if checkpoint_path == None else None
        with tf.summary.FileWriter(args.log_dir, graph=_graph) as test_writer:
            iterations = alparams["iterations"]
            if iterations < 0:
                # Iterate untill all data is consumed
                iterations = np.ceil(len(unlabelled)
                                     / float(alparams["selection_size"]))
                logger.info("Iteration count: %d" % iterations)

            while state["iteration"] < iterations:
                # Step 1: train_loop
                train_input.set_indices(labelled)

                if state["iteration"] == 0:
                    # Pretrain
                    log_subdir = os.path.join(args.log_dir, "pretrain")
                    # Only use labelled subset
                else:
                    # Any other iteration
                    log_subdir = os.path.join(args.log_dir, "iter-%d" %
                                              state["iteration"])
                    # Sample from the unlabelled set
                    p = alparams["pseudo_labelling_proportion"]
                    sample_size = int(len(labelled)*p/(1-p))
                    sample_size = np.minimum(sample_size, len(unlabelled))
                    train_input.set_sample_size(sample_size)

                # Create subdir if it doesn't exist
                if not os.path.exists(log_subdir):
                    os.mkdir(log_subdir)

                # Change checkpoint manager directory
                ckpt_manager.chdir(log_subdir)
                with tf.summary.FileWriter(log_subdir) as train_val_writer:
                    # Enter train loop
                    try:
                        checkpoint_path = train_loop(train_val_writer)
                    except KeyboardInterrupt as exception:
                        # Quickly store state
                        if ckpt_manager.latest_checkpoint != "":
                            state["checkpoint"] = ckpt_manager.latest_checkpoint
                        with open(state_filename, "w") as f:
                            json.dump(state, f, indent=2)
                            f.truncate()
                        raise exception


                # Reload best checkpoint
                status = checkpoint.restore(checkpoint_path)
                status.run_restore_ops(sess)
                sess.run(update_val_op)

                # Step 2: test_loop
                test_loop(test_writer)

                # Step 3: Find low confidence examples
                # Reset train_input to use all examples for ranking
                train_input.set_indices()
                if alparams["selection_size"] > 0:
                    low_conf_examples = rank_confidence()
                else:
                    # Draw examples randomly
                    low_conf_examples = np.random.choice(
                        unlabelled, np.abs(alparams["selection_size"]))

                # (maybe) Pause for user to annotate
                to_annotate_indices = no_label_indices[np.isin(
                    no_label_indices, low_conf_examples)]

                while len(to_annotate_indices) > 0:
                    to_annotate = train_examples[to_annotate_indices]
                    # Poll user for filenames of annotated examples
                    logger.info("Please annotate the following examples:\n%s" %
                                "\n".join(to_annotate_basename.tolist()))
                    filenames = tkinter.filedialog.askopenfilename(
                        multiple=1,
                        filetypes=(("TFRecord", "*.tfrecord"),))

                    hit = [] # List of matching filename indices
                    for filename in filenames:
                        basename = os.path.basename(filename)
                        idx = -1
                        for i in range(len(to_annotate)):
                            if to_annotate[i].endswith(basename):
                                idx = i
                                break
                        if idx != -1:
                            # Update state filenames
                            train_examples[to_annotate_indices[idx]] = filename
                            hit.append(idx)
                        else:
                            logger.info("Unrecognized filepath: %s" % filename)
                    # Remove matched paths
                    to_annotate_indices = np.delete(to_annotate_indices, hit)


                # Remove annotated examples from unlabelled set
                no_label_indices = no_label_indices[np.isin(no_label_indices,
                                                             low_conf_examples,
                                                             invert=True)]


                logger.info(
                    "Moving following examples to labelled set:\n%s" %
                    "\n".join(train_examples[low_conf_examples].tolist())
                )
                # First make the update to input stage before
                # commiting state change
                train_input_labelled[low_conf_examples] = True
                train_input.labelled = train_input_labelled


                # Step 4: Update state information
                labelled = np.append(labelled, low_conf_examples)
                unlabelled = unlabelled[np.isin(unlabelled, low_conf_examples,
                                            assume_unique=True, invert=True)]
                state["dataset"]["train"]["filenames"] = train_examples.tolist()
                state["dataset"]["train"]["labelled"] = labelled.tolist()
                state["dataset"]["train"]["unlabelled"] = unlabelled.tolist()
                state["iteration"] += 1
                state["checkpoint"] = checkpoint_path
                # Dump updated state
                with open(state_filename, "w") as f:
                    json.dump(state, f, indent=2)
                    f.truncate()
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
    :rtype:   argparse.Namespace
    """
    # Required arguments
    req_parser = argparse.ArgumentParser(add_help=False)
    req_group = req_parser.add_argument_group(title="Required arguments")
    req_group.add_argument(
        "-d", "--data-dir",
        required=True,
        type=str,
        dest="data_dir",
        help="Path to dataset root directory"
    )
    req_group.add_argument(
        "-l", "--log-dir",
        required=True,
        type=str,
        dest="log_dir",
        metavar="LOG_DIR",
        help="Logdirectory for the session."
    )
    req_group.add_argument(
        "-p", "--parameters",
        required=True,
        type=str,
        dest="params",
        metavar="PARAM_FILE",
        help="Path to parameter configuration file, see conf subdirectory."
    )
    #Optional arguments
    opt_parser = argparse.ArgumentParser(add_help=False)
    opt_parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        dest="checkpoint", required=False,
        metavar="CHECKPOINT",
        help="Path to pretrained checkpoint directory or model."
    )
    opt_parser.add_argument(
        "-u", "--unlabelled-dir",
        type=str,
        default=None,
        dest="unlabelled",
        metavar="UNLABELLED_GLOB",
        help="Path to directory containing only feature data."
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
    sys.exit(main(args, logger))
