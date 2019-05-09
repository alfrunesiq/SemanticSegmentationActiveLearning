import glob
import json
import logging
import multiprocessing
import os

from functools import partial

import numpy as np
import tensorflow as tf
from google.protobuf.json_format import MessageToJson

from . import tfrecord

_CPU_COUNT = multiprocessing.cpu_count()
del multiprocessing

def generate_mask(labels, mask_index=255):
    """
    Generates a binary mask that is zero for @labels == @mask_index,
    and maps the appropriate labels to zero.

    :param labels:     input labels (ground truth)
    :param mask_index: index to mask out
    :returns: labels with @mask_index masked to zero, mask
    :rtype:   tf.Tensor, tf.Tensor
    """

    _labels = tf.squeeze(labels, axis=-1)
    # Create binary mask
    mask_bool = tf.math.not_equal(_labels, mask_index)
    mask = tf.cast(mask_bool, labels.dtype)
    # Map masked labels to zero
    _labels = tf.where(mask_bool, _labels, mask)

    return _labels, mask


class InputStage:
    """
    Class for handling dataset and input processing.

    The class can contain multiple datasets, and also holds a
    reinitializable iterator that can be used accross the contained
    datasets.
    The contained datasets are supposed to be training-, validation-
    and possibly testset, as it is internally assumed that the output
    matches.
    """
    def __init__(self, input_shape=[512,512], scope="Dataset"):
        """
        Initializes the @InputStage class

        :param record_fmt: dict with TFRecord layout mapping record keys
                           to tf.[Fixed|Var]LenFeature.
        :param input_shape: Input shape of the network
        """
        self.logger     = logging.getLogger(__name__)
        with tf.name_scope(scope) as _scope:
            self.name_scope = _scope

        if len(input_shape) == 3:
            self.shape = input_shape
        elif len(input_shape) == 2:
            self.shape = input_shape + [None]
        else:
            logger.warn("Proceeding with unknown inputshape.")
            self.shape = [None,None,None]

        self.iterator   = None
        self.datasets   = {}

    def add_dataset(self, name, file_patterns, batch_size, epochs=1,
                    parse_fn=None, decode_fn=None, augment=None):
        """
        Add a new dataset to the collection self.datasets dictionary.

        NOTE: All added datasets are assumed to have the same output-
              types. The addition of multiple datasets are mainly for the
              puropose of containint train-, validation- and testset.

        :param name:          dataset identifier used to reference the data-
                              set for retrieving dataset/iterator
        :param file_patterns: file pattern glob to TFRecord files
        :param batch_size:    Batch size for the dataset
        :param parse_fn:      Parses TFRecord file, applied before @decode_fn
        :param decode_fn:     Optional custom parse and decode function to
                              read, parse and decode tfrecord files from
                              input filepath.
        :param augment:       (bool) Wheather to apply augmentation to
                              default decoder. (requires decode_fn=None)
        """
        # make sure path is contained in a list
        if not isinstance(file_patterns, list):
            file_patterns = [file_patterns]
        # Count number of examples
        num_examples = 0
        for i in range(len(file_patterns)):
            if os.path.isdir(file_patterns[i]):
                # Use default glob
                file_patterns[i] = os.path.join(file_patterns[i], "*.tfrecord")
            num_examples += len(glob.glob(file_patterns[i]))

        with tf.name_scope(self.name_scope):
            with tf.name_scope(name):
                # NOTE: tf.data.Dataset.list_files shuffles as well
                dataset = tf.data.Dataset.list_files(file_patterns)
                if epochs != 0:
                    dataset = dataset.repeat(epochs)
                dataset = tf.data.TFRecordDataset(dataset)

                self._add_dataset(name, dataset, batch_size,
                                  parse_fn, decode_fn, augment)

        # Store number of elements in the dataset
        self.datasets[name]["count"] = num_examples

        return num_examples

    def add_dataset_from_placeholders(self, name, filenames,
                                      *aux_placeholders,
                                      batch_size=8, parse_fn=None,
                                      decode_fn=None, augment=None):
        """
        Creates dataset from placeholders.
        DISCLAIMER: this function is callee shuffle; i.e. the dataset is
                    "locked" to 1 epoch, and caller is responsible for
                    shuffling the data at initialization.
        :param filenames: tf.placeholder of 1D array of filenames.
        :param *aux_placeholders: any auxillary placeholders fed parallel
                                  with records dataset.
        :param batch_size: batch size for each iterator call
        :param parse_fn:   Parses TFRecord file, applied before @decode_fn
        :param decode_fn:  (Optional) custom parse and decode function to
                           read, parse and decode tfrecord files from
                           input filepath.
        :param augment:    (bool) Wheather to apply augmentation to
                           default decoder. (requires decode_fn=None)
        """
        num_examples = filenames.shape.as_list()[0] # Might be None
        with tf.name_scope(self.name_scope):
            with tf.name_scope(name):
                dataset = tf.data.TFRecordDataset(filenames,
                                                  buffer_size=256*1024*1024)
                if len(aux_placeholders) > 0:
                    aux_datasets = [tf.data.Dataset.from_tensor_slices(ph) \
                                    for ph in aux_placeholders]
                    dataset = tf.data.Dataset.zip((dataset,) +
                                                  tuple(aux_datasets))

                self._add_dataset(name, dataset, batch_size,
                                  parse_fn, decode_fn, augment)

        self.datasets[name]["count"] = filenames.shape.as_list()[0]
        return num_examples


    def _add_dataset(self, name, dataset, batch_size, parse_fn=None,
                     decode_fn=None, augment=None):
        # Create parse function for extracting a tf.Example from the record
        # Parse TFRecord
        if parse_fn == None:
            def parse_fn(path, *aux):
                # Default parse function
                fmt = {
                    "image/channels":tf.io.FixedLenFeature((),tf.int64,-1),
                    "image/data" : tf.io.FixedLenFeature((),tf.string,""),
                    "label"      : tf.io.FixedLenFeature((),tf.string,""),
                    "height"     : tf.io.FixedLenFeature((),tf.int64 ,-1),
                    "width"      : tf.io.FixedLenFeature((),tf.int64 ,-1)
                }
                ret = tf.io.parse_single_example(
                    path, fmt, name="ParseExample"), *aux
                return ret

        dataset = dataset.map(parse_fn,
                              num_parallel_calls=_CPU_COUNT-1)
        # Decode images
        if decode_fn == None:
            if augment == True:
                decoder = partial(self.default_decoder,augment=True)
                dataset = dataset.map(decoder,
                                      num_parallel_calls=_CPU_COUNT-1)
            else:
                dataset = dataset.map(self.default_decoder,
                                      num_parallel_calls=_CPU_COUNT-1)
        else:
            dataset = dataset.map(decode_fn,
                                  num_parallel_calls=_CPU_COUNT-1)

        # Batch and prefetch image/label data
        if batch_size > 1:
            dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(batch_size)
        self.datasets[name] = {}
        # Store dataset
        self.datasets[name]["dataset"]  = dataset
        # Create [re-]initializable iterator
        if self.iterator == None:
            self.iterator = tf.data.Iterator.from_structure(
                output_types=dataset.output_types,
                output_shapes=dataset.output_shapes,
                output_classes=dataset.output_classes
            )
        # Add initialization op for this datset
        self.datasets[name]["init"] = \
            self.iterator.make_initializer(dataset)

    def get_datset(self, name):
        return self.datasets[name]["dataset"]

    def init_iterator(self, name, sess, feed_dict=None):
        """
        Convenience function to initialize the iterator for the dataset
        referenced by @name.
        NOTE: iterating past the dataset raises tf.errors.OutOfRangeError
        :param name:      dataset identifier specified when @add_dataset
                          was called
        :param sess:      an active tf.Session to run the initializer
        :param feed_dict: (Optional) feed_dict if dataset contains
                          placeholders
        """
        # Fetch the iterator
        init_op = self.datasets[name]["init"]
        # Run the initializer
        sess.run(init_op, feed_dict)

    def get_output(self):
        """
        NOTE: need to run initializer
        """
        return self.iterator.get_next()

    def default_decoder(self, example, *other_outputs, augment=False):
        """
        Decodes all images in example and stacks all channels along with
        the decoded label image in the last channel.

        :param example: parsed example (tf.Example)
        :returns: Tensors of image, label and mask bypassing all
                  additional upstream parser outputs.
        :rtype:   tf.Tensor
        """
        # Decode image and label from raw record data
        image = tf.image.decode_image(example["image/data"],
                                      dtype=tf.uint8,
                                      name="DecodeImage")
        label = tf.cond(tf.math.not_equal(example["label"], ""),
                        true_fn=lambda: tf.image.decode_png(example["label"],
                                                            channels=1,
                                                            dtype=tf.uint8,
                                                            name="DecodeLabel"),
                        false_fn=lambda: tf.fill(
                            dims=[example["height"], example["width"], 1],
                            value=tf.constant(255, dtype=tf.uint8,
                                              name="Fill/value"),
                            name="NoLabel"),
                        name="Label"
        )
        channels = 3
        # Iterate over example keys and collect all images not label
        self.shape[-1] = channels
        # Stack the images' channels and label image
        image_stack = tf.concat([image, label], axis=2, name="StackImages")
        # Need to recover channel dimension
        stack_shape = image_stack.shape.as_list()
        stack_shape[-1] = self.shape[-1]+1
        image_stack.set_shape(stack_shape)

        if augment:
            # NOTE extra argument @image_dist
            image, image_dist, label, mask = \
                    self._default_augmentation(image_stack)
            ret = image, image_dist, label, mask, *other_outputs
        else:
            # TODO: move this to a separate function "_default_no_augmentation"
            # Height and width need not be the same accross samples
            center = [tf.cast(example["height"]//2, tf.int32),
                      tf.cast(example["width"]//2, tf.int32)]
            top_left = [center[0]-self.shape[0]//2,
                        center[1]-self.shape[1]//2]
            crop = tf.image.crop_to_bounding_box(image_stack,
                                                 top_left[0], top_left[1],
                                                 self.shape[0], self.shape[1])
            # retrieve image and label channels from crop
            image = crop[:,:,:self.shape[-1]]
            label = crop[:,:,self.shape[-1]:]
            # Convert image dtype and normalize
            image = tf.image.convert_image_dtype(image, tf.float32,
                                                 name="ToFloat")
            # Generate mask of valid labels (and squeeze unary 3rd-dim)
            label, mask = generate_mask(label)
            ret = image, label, mask, *other_outputs
        return ret

    def _default_augmentation(self, images):
        """
        Applies random crop, flip and channel scaling and separates
        image from labels.
        :param images: image channels and labels stacked
        :returns: images   , label
        :rtype:   tf.Tensor, tf.Tensor
        """
        with tf.name_scope("DataAugmentation"):
            channels   = self.shape[-1]
            crop_shape = self.shape
            crop_shape[-1] += 1
            # Random channelwise pixel intensity scaling
            px_scaling = tf.random.uniform(shape=[channels], maxval=1.4,
                                           minval=0.8, dtype=tf.float32)
            # Crop out a sizable portion randomly
            image_crop = tf.random_crop(images, crop_shape,
                                        name="RandomCrop")
            image_crop = tf.image.random_flip_left_right(image_crop)
            # Slice out the image stack and label
            image = image_crop[:,:,:channels] # [h,w,c]
            label = image_crop[:,:,channels:] # [h,w,1]
            # NOTE this also scales values to range [0.0, 1.0]
            image = tf.image.convert_image_dtype(image, tf.float32,
                                                 name="ToFloat")

            # Apply random channel scaling
            image_dist = tf.multiply(image, px_scaling, 
                                     name="RandomPixelScaling")
            image_dist = tf.clip_by_value(image_dist, 0.0, 1.0,
                                          name="RandomPixelScaling/Clip")
            # Generate mask
            label, mask = generate_mask(label)
        return image, image_dist, label, mask

class NumpyCapsule:
    def __init__(self, shuffle=True):
        # Initialize internal state
        self._setattr = True    # __setattr__ toggle ndarray tracking
        self._feed_dict = {}    # internal feed_dict
        self._placeholders = {} # placeholder dict map
        self.shuffle = shuffle  # whether to shuffle feed_dict property
        self._length = 0        # length tracker
        self._cur_length = 0    # current length (changes on @set_indices)
        self._indices = []      # indices of values used in feed_dict
        self._full_range = []   # all possible indices
        self._sample_set = []   # treat the difference of @_indices and
                                # @_full_range as sample set
        self._sample_size = 0
        self._sample_prob = None

    @property
    def feed_dict(self):
        feed_dict = {}
        if self.shuffle:
            # Shuffles all values before returning
            indices = self._indices.copy()
            if self._sample_size > 0:
                rand_indices = np.random.choice(self._sample_set, 
                                                self._sample_size,
                                                replace=False,
                                                p=self._sample_prob)
                indices = np.concatenate((indices, rand_indices))
            np.random.shuffle(indices)
            for name in self._placeholders:
                feed_dict[self._placeholders[name]] = \
                    self._feed_dict[self._placeholders[name]][indices]
        else:
            for name in self._placeholders:
                feed_dict[self._placeholders[name]] = \
                        self._feed_dict[self._indices]
        return feed_dict

    def set_indices(self, indices=None, sample_indices=None, sample_prob=None):
        """
        Manually set indeces to use subset of values.
        :param indices: indices set of values to be used.
        """
        try:
            self._setattr = False
            if indices is None:
                self._indices = self._full_range
                self._cur_length = self._length
                self._sample_set = []
                self._sample_size = 0
                self._sample_prob = None
            else:
                self._indices = indices
                self._cur_length = len(self._indices)
                if sample_indices is None:
                    self._sample_set = self._full_range[np.isin(self._full_range, 
                                                                self._indices, 
                                                                invert=True)]
                else:
                    self._sample_set = sample_indices
                    if sample_prob is not None:
                        if len(sample_prob) == len(self._sample_set):
                            self._sample_prob = sample_prob
        finally:
            self._setattr = True

    def set_sample_size(self, size):
        self._sample_size = size
        return self._sample_size

    def get_value(self, attribute):
        """
        Get value of the entry containing attribute.
        :param attribute: attribute (placeholder) handle.
        """
        return self._feed_dict[attribute]

    @property
    def size(self):
        return self._cur_length + self._sample_size

    def __setattr__(self, name, value):
        if isinstance(value, np.ndarray) and not name.startswith("_"):
            if name in self._placeholders:
                # Cache array
                self._feed_dict[name] = value
            else:
                # Create placeholder
                self._placeholders[name] = tf.placeholder(
                    value.dtype,
                    shape=[None]*len(value.shape),
                    name=name)
                # Create feed_dict entry
                self._feed_dict[self._placeholders[name]] = value

            # NOTE: User is required to keep track of equal length arrays
            if self._length != len(value):
                self._length = len(value)
                # Update internal indeces
                self._full_range = np.arange(self._length)
                self.set_indices()
            # Actually set attribute to placeholder
            super(NumpyCapsule, self).__setattr__(name, self._placeholders[name])
        else:
            super(NumpyCapsule, self).__setattr__(name, value)



def _peek_tfrecord(filepath):
    """
    Parses a single example tfrecord, and returns a dictionary of the
    first tf.train.Feature
    WARNING tfrecord format and TFRecordWriter may change in future
            Tensorflow releases
    :param filepath: path to a tfrecord containing examples
    :returns: first feature in the record
    :rtype:   dict
    """
    example = tfrecord.tfrecord2tfexamples(filepath)
    # Convert example to dict
    example_dict = tfrecord.MessageToDict(example)

    # Strip off redundant keys
    fmt = example_dict["features"]["feature"]
    # Strip off raw byte data
    def _strip_byteslists(d):
        for k in d.keys():
            if k == "bytesList":
                del d[k]["value"]
            elif isinstance(d[k], dict):
                _strip_byteslists(d[k])
        return d
    return _strip_byteslists(fmt)

def _example_from_format(fmt):
    logger = logging.getLogger(__name__)
    tf_fmt = {}
    for k in fmt.keys():
        # feature-type is given as the first (and only) key in fmt[k]
        _type = str(next(iter(fmt[k])))
        if _type == "bytesList":
            tf_fmt[k] = tf.io.FixedLenFeature((), tf.string, "")
        elif _type == "int64List":
            tf_fmt[k] = tf.io.FixedLenFeature((), tf.int64 , -1)
        else:
            logger.warn("Could not resolve TFRecord format key %s" % _type)
    return tf_fmt

__all__ = ["InputStage", "generate_mask"]
