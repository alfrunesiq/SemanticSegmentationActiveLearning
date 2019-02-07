import tensorflow as tf

class Eval:
    """
    Class for evaluating the classification results of the network.
    """
    def __init__(self, logits, labels, num_classes, mask):
        self.nclasses = num_classes

        with tf.name_scope("Eval"):
            # Convert logits to predictions
            pred   = tf.math.argmax(logits, axis=-1,
                                    output_type=tf.int32,
                                    name="Predictions")
            # Flatten labels and predictions
            labels_ = tf.reshape(labels, [-1])
            pred_   = tf.reshape(pred  , [-1])
            mask_   = tf.reshape(mask  , [-1])

            # Create global confusion matrix (should be reset every epoch)
            self.confusion_mat = tf.get_variable(
                name="ConfusionMat", shape=[self.nclasses, self.nclasses],
                initializer=tf.initializers.zeros(), dtype=tf.int64,
                trainable=False)
            self.batch_confusion_mat = tf.confusion_matrix(
                labels_, pred_, self.nclasses, weights=mask_,
                dtype=tf.int64, name="BatchConfusionMat")
            self.confusion_mat_update = tf.assign_add(self.confusion_mat,
                                                      self.batch_confusion_mat,
                                                      name="ConfusionMatUpdate")
            # Create batch and global level metric ops
            self.metrics       = self._create_metrics(self.confusion_mat,
                                                      scope="Metrics")
            self.batch_metrics = self._create_metrics(self.batch_confusion_mat,
                                                      scope="BatchMetrics")
            self.summaries = None


    def get_metrics_update_op(self):
        """
        Gets the update op for accumulating the confusion matrix.
        :returns: the tensor output of the update operation
        :rtype:   tf.Tensor

        """
        return self.confusion_mat_update

    def reset_metrics(self, sess):
        """
        Resets the confusion matrix accumulator
        :param sess: tf.Session to run the initializer
        """
        sess.run(self.confusion_mat.initializer)

    def get_summaries(self):
        """
        Evaluation metric summaries.
        Organizes the summaries in a dict as follows:
        {
            "BatchMetrics/Metrics": {
                "ConfusionMat" : confusion_summary,
                "Global"       : global_summary,
                "Class"        : per_class_summary
            }
        }
        Where the Batch metrics are evaluated for the current batch,
        where as the "Metrics" entry is evaluated based on the
        accumulated confusion matrix (which is supposed to be
        accumulated [see @get_metrics_update_op] evaluated on a per-epoch
        basis, and correspondingly reset after evaluation).
        :returns: the summaries
        :rtype:   dict

        """
        if self.summaries != None:
            return self.summaries

        self.summaries = {}
        # Walk over metrics
        with tf.name_scope("Summary"):
            for scope in ["BatchMetrics", "Metrics"]:
                with tf.name_scope(scope):
                    per_class_summaries = []
                    metric_dict = \
                        self.batch_metrics if scope == "BatchMetrics" \
                                           else self.metrics
                    confusion_mat = \
                        self.batch_confusion_mat if scope == "BatchMetrics" \
                                                 else self.confusion_mat
                    # Create per-class summaries
                    for i in range(self.nclasses):
                        class_acc  = tf.summary.scalar(
                            "Class_%d_Accuracy" % i,
                            metric_dict["ClassAccuracy"][i])
                        class_prec = tf.summary.scalar(
                            "Class_%d_Precission" % i,
                            metric_dict["ClassPrecission"][i])
                        class_rec  = tf.summary.scalar(
                            "Class_%d_Recall" % i,
                            metric_dict["ClassRecall"][i])
                        class_miou = tf.summary.scalar(
                            "Class_%d_MeanIoU" % i,
                            metric_dict["ClassMeanIoU"][i])
                        per_class_summaries.append([class_acc, class_prec,
                                                    class_rec, class_miou])
                    # Merge all per-class summaries to one operation
                    per_class_summary = tf.summary.merge(per_class_summaries,
                                                         name="ClassMetrics")
                    # Create and merge global summaries
                    pix_acc    = tf.summary.scalar(
                        "PixelAccuracy", metric_dict["PixelAccuracy"])
                    mean_iou   = tf.summary.scalar(
                        "MeanIoU", metric_dict["MeanIoU"]
                    )
                    global_summary = tf.summary.merge([pix_acc, mean_iou],
                                                      name="GlobalMetrics")
                    # Create confusion matrix tensor summary
                    confusion_summary = tf.summary.tensor_summary(
                        "ConfusionMatrix", confusion_mat)
                    self.summaries[scope] = {
                        "ConfusionMat" : confusion_summary,
                        "Global"       : global_summary,
                        "Class"        : per_class_summary}
        return self.summaries

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

            # Per-class true positive rate
            TP = tf.linalg.diag_part(confusion_mat, name="TruePositive")

            confusion_off_diag = confusion_mat - TP
            # Per-class false positive rate
            FP = tf.reduce_sum(confusion_off_diag, axis=0, name="FalsePositive")
            # Per-class false negative rate
            FN = tf.reduce_sum(confusion_off_diag, axis=1, name="FalseNegative")

            # NOTE: precission denominator
            TPpFP = TP + FP
            TPpFPpFN = TPpFP + FN
            # Per-class true negative rate
            TN = tf.math.subtract(samples_tot, TPpFPpFN, name="TrueNegative")

            TPpTN = TP + TN
            TPpFN = TP + FN
            # Per-class derived metrics
            class_accuracy   = tf.math.truediv(TPpTN, samples_tot,
                                               name="ClassAccuracy")
            class_precission = tf.math.truediv(TP, TPpFP,
                                               name="ClassPrecission")
            class_recall     = tf.math.truediv(TP, TPpFN,
                                               name="ClassRecall")
            class_miou       = tf.math.truediv(TP, TPpFPpFN,
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
        return metrics
