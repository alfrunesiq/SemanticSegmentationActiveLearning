import numpy as np
from tensorflow.compat import v1 as tf

class Metrics:
    """
    Class for evaluating the classification results of the network.
    """
    def __init__(self, predictions, labels, num_classes, mask, scope="Metrics"):
        self.nclasses = num_classes

        with tf.variable_scope("Eval") as _scope:
            # Create global confusion matrix (should be reset every epoch)
            self._confusion_mat = tf.Variable(
                tf.zeros(shape=[self.nclasses, self.nclasses],
                         dtype=tf.int64),
                name="ConfusionMat", trainable=False)
            batch_confusion_mat = confusion_mat(
                labels, predictions, self.nclasses, weights=mask,
                dtype=tf.int32, name="BatchConfusionMat")
            self._confusion_mat_update_op = tf.assign_add(
                self._confusion_mat, tf.cast(batch_confusion_mat, tf.int64),
                name="ConfusionMatUpdate")
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,
                                 self._confusion_mat_update_op)
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
                              weights=None, dtype=tf.int32, name="ConfusionMat"):
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

    def get_update_op(self):
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
                    metrics["ClassAccuracy"][i],
                    family="ClassMetrics")
                class_prec = tf.summary.scalar(
                    "Class_%d_Precission" % i,
                    metrics["ClassPrecission"][i],
                    family="ClassMetrics")
                class_rec  = tf.summary.scalar(
                    "Class_%d_Recall" % i,
                    metrics["ClassRecall"][i],
                    family="ClassMetrics")
                class_miou = tf.summary.scalar(
                    "Class_%d_MeanIoU" % i,
                    metrics["ClassMeanIoU"][i],
                    family="ClassMetrics")
                per_class_summaries.append([class_acc, class_prec,
                                            class_rec, class_miou])
            # Merge all per-class summaries to one operation
            per_class_summary = tf.summary.merge(per_class_summaries,
                                                 name="ClassMetrics")
            # Create and merge global summaries
            pix_acc    = tf.summary.scalar(
                "PixelAccuracy", metrics["PixelAccuracy"], family="Metrics")
            mean_iou   = tf.summary.scalar(
                "MeanIoU", metrics["MeanIoU"], family="Metrics"
            )
            metrics_summary = tf.summary.merge([pix_acc, mean_iou],
                                              name="GlobalMetrics")
            # Create confusion matrix tensor summary
            # FIXME This can be done better !
            confusion_summary = tf.summary.text(
                "ConfusionMatrix", tf.as_string(confusion_mat))
            summaries = {
                "ConfusionMat" : confusion_summary,
                "Metrics"      : metrics_summary,
                "ClassMetrics" : per_class_summary}
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

            # Per-class true positive (NOTE: diag_part has no GPU kernel)
            # TP = tf.linalg.diag_part(confusion_mat, name="TruePositive")
            eyes = tf.constant(np.eye(self.nclasses), dtype=confusion_mat.dtype)
            TP_diag = tf.math.multiply(confusion_mat, eyes)
            TP = tf.reduce_sum(TP_diag, axis=1, name="TruePositive")

            confusion_off_diag = confusion_mat - TP_diag
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

def confusion_mat(labels, predictions, num_classes,
                          weights=None, dtype=tf.int32, name="ConfusionMat"):
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
        _labels       = tf.cast(labels, tf.int32)
        _labels_      = tf.reshape(_labels,[-1])
        _predictions  = tf.cast(predictions, tf.int32)
        _predictions_ = tf.reshape(_predictions,[-1])
        if weights is not None:
            _weights  = tf.cast(weights, dtype)
            _weights_ = tf.reshape(_weights,[-1])
        else:
            _weights_ = None
        shape = [num_classes,num_classes]
        flat_shape = num_classes*num_classes
        conf_mat_ = tf.math.bincount(num_classes*_labels_ + _predictions_, 
                                     weights=_weights_, 
                                     minlength=flat_shape, 
                                     maxlength=flat_shape, 
                                     dtype=dtype)
        confusion_mat = tf.reshape(conf_mat_, shape, name="ConfusionMat")
    return confusion_mat
__all__ = ["Eval", "confusion_mat"]
