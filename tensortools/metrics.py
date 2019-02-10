import tensorflow as tf

class Eval:
    """
    Class for evaluating the classification results of the network.
    """
    def __init__(self, predictions, labels, num_classes, mask):
        self.nclasses = num_classes

        with tf.variable_scope("Eval"):
            # Flatten labels and predictions
            labels_ = tf.reshape(labels     , [-1])
            pred_   = tf.reshape(predictions, [-1])
            mask_   = tf.reshape(mask       , [-1])

            # Create global confusion matrix (should be reset every epoch)
            self._confusion_mat = tf.get_variable(
                name="ConfusionMat", shape=[self.nclasses, self.nclasses],
                initializer=tf.initializers.zeros(), dtype=tf.int64,
                trainable=False)
            batch_confusion_mat = self._intrnl_confusion_mat(
                labels_, pred_, self.nclasses, weights=mask_,
                dtype=tf.int32, name="BatchConfusionMat")
            self._confusion_mat_update_op = tf.assign_add(
                self._confusion_mat, tf.cast(batch_confusion_mat, tf.int64),
                name="ConfusionMatUpdate")
            self._confusion_mat_reset_op = self._confusion_mat.initializer
            # Create batch and global level metric ops
            # NOTE: confusion_mat gets entry "ConfusionMat" in the returned dict
            self.metrics       = self._create_metrics(self._confusion_mat,
                                                      scope="Metrics")
            self.batch_metrics = self._create_metrics(batch_confusion_mat,
                                                      scope="BatchMetrics")
            self.summaries = None
            self.batch_summaries  = None

    def _intrnl_confusion_mat(self, labels, predictions, num_classes,
                              weights=None, dtype=tf.int32, name=None):
        """
        Had to reimplement the tf.confusion_matrix function as the
        unneeded assertion operators takes an unnecessary amount of time.
        :param labels:      1D tensor of ground trurth labels
        :param predictions: 1D tensor of predictions (argmax)
        :param num_classes: total number of classes
        :param dtype:       dtype of the returned values
        :param name:        name of the scope of the operations
        :returns: Confusion matrix
        :rtype:   tf.Tensor

        """
        with tf.name_scope(name):
            _labels      = tf.dtypes.cast(labels, tf.int64)
            _predictions = tf.dtypes.cast(predictions, tf.int64)
            _weights     = weights
            if _weights is not None:
                _weights     = tf.dtypes.cast(weights, dtype)
            shape = [num_classes,num_classes]
            indices = tf.stack([_labels, _predictions], axis=1)
            values = tf.ones_like(_labels, dtype=dtype) if weights is None \
                     else _weights
            confusion_mat_sparse = tf.sparse.SparseTensor(indices=indices,
                                                          values=values,
                                                          dense_shape=shape)
            zeros = tf.zeros(shape, dtype=dtype)
        return tf.sparse.add(confusion_mat_sparse, zeros)

    def get_metrics_update_op(self):
        """
        Gets the update op for accumulating the confusion matrix.
        :returns: the tensor output of the update operation
        :rtype:   tf.Tensor
        """
        return self._confusion_mat_update_op

    def reset_metrics(self, sess):
        """
        Resets the confusion matrix accumulator
        :param sess: tf.Session to run the initializer
        """
        sess.run(self._confusion_mat.initializer)

    def get_batch_summaries(self):
        if self.batch_summaries == None:
            self.batch_summaries = self._create_summaries(
                self.batch_metrics, scope="BatchMetrics")
        return self.batch_summaries

    def get_summaries(self):
        if self.summaries == None:
            self.summaries = self._create_summaries(self.metrics,
                                                    scope="Metrics")
        return self.summaries

    def _create_summaries(self, metrics, scope=None):
        """
        Evaluation metric summaries.
        Organizes the summaries in a dict as follows:
        {
            "ConfusionMat" : confusion_summary,
            "Global"       : global_summary,
            "Class"        : per_class_summary
        }
        Where the Batch metrics are evaluated for the current batch,
        where as the "Metrics" entry is evaluated based on the
        accumulated confusion matrix (which is supposed to be
        accumulated [see @get_metrics_update_op] evaluated on a per-epoch
        basis, and correspondingly reset after evaluation).
        :returns: the summaries
        :rtype:   dict
        """
        summaries = None
        # Walk over metrics
        with tf.name_scope(scope):
            per_class_summaries = []
            confusion_mat = metrics["ConfusionMat"]
            # Create per-class summaries
            for i in range(self.nclasses):
                class_acc  = tf.summary.scalar(
                    "Class_%d_Accuracy" % i,
                    metrics["ClassAccuracy"][i])
                class_prec = tf.summary.scalar(
                    "Class_%d_Precission" % i,
                    metrics["ClassPrecission"][i])
                class_rec  = tf.summary.scalar(
                    "Class_%d_Recall" % i,
                    metrics["ClassRecall"][i])
                class_miou = tf.summary.scalar(
                    "Class_%d_MeanIoU" % i,
                    metrics["ClassMeanIoU"][i])
                per_class_summaries.append([class_acc, class_prec,
                                            class_rec, class_miou])
            # Merge all per-class summaries to one operation
            per_class_summary = tf.summary.merge(per_class_summaries,
                                                 name="ClassMetrics")
            # Create and merge global summaries
            pix_acc    = tf.summary.scalar(
                "PixelAccuracy", metrics["PixelAccuracy"])
            mean_iou   = tf.summary.scalar(
                "MeanIoU", metrics["MeanIoU"]
            )
            global_summary = tf.summary.merge([pix_acc, mean_iou],
                                              name="GlobalMetrics")
            # Create confusion matrix tensor summary
            # FIXME This can be done better !
            confusion_summary = tf.summary.text(
                "ConfusionMatrix", tf.as_string(confusion_mat))
            summaries = {
                "ConfusionMat" : confusion_summary,
                "Global"       : global_summary,
                "Class"        : per_class_summary}
        return summaries

    def _create_metrics(self, confusion_mat, scope=None):
        """
        Derives evaluation metrics from the confusion matrix

        Consider 3-class case
        [ [ [ TP , FN , FN ], [ [ TN , FP , TN ], [ [ TN , TN , FP ],
            [ FP , TN , TN ],   [ FN , TP , FN ],   [ TN , TN , FP ],
            [ FP , TN , TN ] ]  [ TN , FP , TN ] ]  [ FN , FN , TP ] ] ]

        :param confusion_mat: confusion matrix
        :param scope:         scope for the operations
        :returns: fundamental and derived evaluation matrics
        :rtype:   dict

        """
        # Evaluation metrics can be evaluated directly from
        # the confusion matrix:
        # Consider 3 classes:
        with tf.name_scope(scope):
            # Get total samples (pixels) accumulated in confusion mat
            samples_tot = tf.reduce_sum(confusion_mat)

            # Per-class true positive
            TP = tf.linalg.diag_part(confusion_mat, name="TruePositive")

            confusion_off_diag = confusion_mat - tf.linalg.diag(TP)
            # Per-class false positives
            FP = tf.reduce_sum(confusion_off_diag, axis=0, name="FalsePositive")
            # Per-class false negatives
            FN = tf.reduce_sum(confusion_off_diag, axis=1, name="FalseNegative")

            # NOTE: precission denominator
            TPpFP = TP + FP
            TPpFPpFN = TPpFP + FN
            # Per-class true negatives
            TN = tf.math.subtract(samples_tot, TPpFPpFN, name="TrueNegative")

            TPpTN = TP + TN
            TPpFN = TP + FN
            # Per-class derived metrics
            class_accuracy   = tf.math.truediv(TPpTN, samples_tot,
                                               name="ClassAccuracy")
            class_precission = tf.math.truediv(TP, tf.math.maximum(TPpFP, 1),
                                               name="ClassPrecission")
            class_recall     = tf.math.truediv(TP, tf.math.maximum(TPpFN, 1),
                                               name="ClassRecall")
            class_miou       = tf.math.truediv(TP, tf.math.maximum(TPpFPpFN, 1),
                                               name="ClassMeanIoU")

            # Global derived metrics
            pix_accuracy = tf.math.truediv(tf.reduce_sum(TP), samples_tot,
                                           name="PixelAccuracy")
            mean_iou = tf.reduce_mean(class_miou, name="MeanIoU")

        metrics = {}
        metrics["TruePositive"]    = TP
        metrics["TrueNegative"]    = TN
        metrics["FalsePositive"]   = FP
        metrics["FalseNegative"]   = FN
        metrics["ClassAccuracy"]   = class_accuracy
        metrics["ClassPrecission"] = class_precission
        metrics["ClassRecall"]     = class_recall
        metrics["ClassMeanIoU"]    = class_miou
        metrics["PixelAccuracy"]   = pix_accuracy
        metrics["MeanIoU"]         = mean_iou
        metrics["ConfusionMat"]    = confusion_mat
        return metrics

__all__ = ["Eval"]
