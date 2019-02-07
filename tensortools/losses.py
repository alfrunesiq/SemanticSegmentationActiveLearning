import tensorflow as tf

def masked_softmax_cross_entropy(
        labels,
        logits,
        num_classes,
        mask_index=255,
        weighted_mask_eps=0.0,
        label_smoothing=0.0,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        scope=None):

    mask = None
    with tf.name_scope(scope):
        # FIXME make mask an argument, and alternatively create a @weights variable
        # Squeeze out that singleton dimension
        if labels.shape.ndims == 4:
            _labels = tf.squeeze(labels, axis=3)
        if weighted_mask_eps > 0.0:
            # Start off with a zero mask and incrementally add in each
            # label weighted by the inverse of it's respective importance
            # NOTE mask index is automatically ignored (but assumes
            #      label range [0,@num_classes])
            mask = tf.zeros_like(_labels, dtype=tf.float32)
            for i in range(num_classes):
                _label_mask   = tf.cast(tf.equal(_labels, i), tf.float32)
                _label_weight = 1.0/(tf.reduce_mean(_label_mask)
                                     + weighted_mask_eps)
                mask = mask + _label_mask * _label_weight
        else:
            # Create mask to ignore @mask_index labels
            mask = tf.cast(tf.math.not_equal(_labels, mask_index),
                           dtype=tf.float32)
        # Map @mask_index to 0 (does not matter as long as label is valid)
        zeros   = tf.zeros_like(_labels)
        _labels = tf.where(tf.equal(_labels,mask_index), zeros, _labels)
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

