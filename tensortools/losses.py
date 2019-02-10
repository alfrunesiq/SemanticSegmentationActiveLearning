import tensorflow as tf

def masked_softmax_cross_entropy(
        labels,
        logits,
        mask,
        num_classes,
        weight=0.0,
        label_smoothing=0.0,
        scope=None):
    """
    Evaluates the softmax cross-entropy loss, and masks out labels using
    @mask parameter. Optionally if @weight is greater than zero, the
    loss also punishes missclassified labels by a factor (1+weights),
    this is to help cope with sparse classes.
    :param labels:          ground truth labels (tf.uint8)
    :param logits:          segmentation output logits
    :param mask:            mask for labels should be ignored
    :param num_classes:     number of valid calsses
    :param weight:          [optional] weight parameter to punish errors
    :param label_smoothing: [optional] label smoothing on per pixel
                            classification
    :param scope:           [optional] scope for the operations
    :returns: The mean cross entropy loss
    :rtype:   tf.Tensor: @logits.dtype (tf.float32)
    """
    with tf.name_scope(scope):
        # Squeeze out that singleton dimension
        _labels = labels
        if labels.shape.ndims == 4:
            _labels = tf.squeeze(_labels, axis=3)

        if weight > 0.0:
            # Start off with a zero mask and incrementally add in each
            # label weighted by the inverse of it's respective importance
            pred = tf.math.argmax(logits, axis=3, name="Predictions")
            error_mask = tf.cast(tf.math.not_equal(pred, _labels), tf.float32)
            _mask = tf.to_float(mask)
            _mask = _mask*(1 + error_mask*weight)
        else:
            # Create mask to ignore @mask_index labels
            _mask = tf.to_float(mask)
        # Apply label smoothing
        on_value  = 1.0 - label_smoothing
        off_value = label_smoothing / (num_classes - 1.0)
        # Generate one-hot labels
        _labels   = tf.one_hot(_labels, num_classes,
                               on_value, off_value,
                               axis=3,
                               dtype=tf.float32,
                               name="LabelsOneHot")
        _labels = tf.stop_gradient(_labels)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=_labels, logits=logits, dim=-1, name="SoftmaxCrossEntropy")
        # Apply mask / weighting
        loss = tf.math.multiply(loss, _mask)
        # Do the mean in two takes: first in batch dimension,
        # then spatial with higher precission
        loss = tf.reduce_mean(loss, axis=0, name="BatchMeanCrossEntropy")
        # Cast to float64, spatial dimensions can make the numeric
        # errors can be severe for high resolution images
        loss = tf.cast(loss, tf.float64)
        # Compute scalar mean loss
        loss = tf.math.reduce_mean(loss, name="MeanCrossEntropy")
    return loss

def L2_regularization(kernels, weight, scope=None):
    """
    Creates scalar L2_regularization loss over the list of @kernels
    :param kernels: list of tf.Tensor kernels
    :param weight:  weight parameter
    :param name:    name / scope of the operation
    :returns: l2 loss over the kernels
    :rtype:   tf.Tensor scalar
    """
    l2_losses = []
    for krnl in kernels:
        # Create name under kernel variable scope
        _name = krnl.name + "/L2Regularization"
        # Append loss (@tf.nn.l2_loss returns a scalar)
        l2_losses.append(tf.nn.l2_loss(krnl, name=_name))
    # Sum over all scalars
    _weight = weight / float(len(kernels))
    with tf.name_scope(scope):
        l2_loss = tf.math.add_n(l2_losses)
        l2_loss = tf.math.multiply(l2_loss, _weight)
    return l2_loss

__all__ = ["masked_softmax_cross_entropy", "L2_regularization"]
