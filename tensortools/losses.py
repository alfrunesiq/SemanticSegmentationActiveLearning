import tensorflow as tf

def masked_softmax_cross_entropy(
        labels,
        logits,
        num_classes,
        mask_index=255,
        label_smoothing=0.0,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        scope=None):

    mask = None
    with tf.name_scope(scope):
        # Squeeze out that singleton dimension
        _labels = tf.squeeze(labels, axis=3)
        # Create mask to ignore @mask_index labels
        ones    = tf.ones_like(_labels)
        zeros   = tf.zeros_like(_labels)
        mask    = tf.where(_labels == mask_index, zeros, ones)
        # Map @mask_index to 0 (does not matter as long as label is valid)
        _labels = tf.where(_labels == mask_index, zeros, _labels)
        # Apply label smoothing
        on_value  = 1.0 - label_smoothing
        off_value = label_smoothing / (num_classes - 1.0)
        # Generate one-hot labels
        _labels   = tf.one_hot(_labels, num_classes,
                               on_value, off_value,
                               axis=3,
                               name="LabelsOneHot")
    loss = tf.losses.softmax_cross_entropy(_labels, logits,
                                           weights=mask,
                                           reduction=reduction,
                                           scope=scope)
    return loss

