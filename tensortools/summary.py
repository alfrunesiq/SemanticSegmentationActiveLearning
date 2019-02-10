import tensorflow as tf

# NOTE: tf.metric is for whole datasets
def get_metric_summaries(logits, label, label_mask):
    pred = tf.argmax(logits)
    TP = tf.equal(pred, label)
    FP = tf.not_equal(pred, label)

