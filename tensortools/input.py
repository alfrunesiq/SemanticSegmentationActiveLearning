import glob
import json
import logging
import multiprocessing
import os

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
    def __init__(self, batch_size,
                 input_shape=[512,512]):
        """
        Initializes the @InputStage class

        :param record_fmt: dict with TFRecord layout mapping record keys
                           to tf.[Fixed|Var]LenFeature.
        :param batch_size:  Batch size for the dataset pipeline
        :param input_shape: Input shape of the network
        """
        self.logger     = logging.getLogger(__name__)
        self.name_scope = tf.name_scope("Dataset")

        if len(input_shape) == 3:
            self.shape = input_shape
        elif len(input_shape) == 2:
            self.shape = input_shape + [None]
        else:
            logger.warn("Proceeding with unknown inputshape.")
            self.shape = [None,None,None]

        self.batch_size = batch_size
        self.datasets   = {}

    def add_dataset(self, name, file_patterns, epochs=1, augment=None, decode_fn=None):
        """
        Add a new dataset to the collection self.datasets dictionary.

        NOTE: All added datasets are assumed to have the same output-
              types. The addition of multiple datasets are mainly for the
              puropose of containint train-, validation- and testset.

        :param name:          dataset identifier used to reference the data-
                              set for retrieving dataset/iterator
        :param file_patterns: file pattern glob to TFRecord files
        :param augment:       Can either be
                              a callable : this function is mapped after the
                                           @decode_fn
                              None       : leaves the images as is
                              other      : applies default augmentation
        :param decode_fn:     Optional custom decode function to decode the
                              parsed TFRecord examples. The function takes a
                              parsed example as argument.
        """
        # make sure path is contained in a list
        if not isinstance(file_patterns, list):
            file_patterns = [file_patterns]

        # Count number of examples
        num_examples = 0
        for i in range(len(file_patterns)):
            num_examples += len(glob.glob(pattern))

        # Peek into a tfrecord to retrieve format and channel count
        _record = next(glob.iglob(file_patterns[0]))
        fmt = _peek_tfrecord(os.path.join(path[0],filenames[0]))
        self.input_depths = {}

        # Extract channel info for the input data
        for key in fmt.keys():
            _key = str(key)
            scope = "/".join(_key.split("/")[:-1])
            if _key.endswith("channels"):
                self.input_depths[scope] = int(fmt[key]["int64List"]["value"][0])

        # Create format specification from the fmt dictionary
        tf_fmt = _example_from_format(fmt)
        # Create parse function for extracting a tf.Example from the record
        with self.name_scope:
            with tf.name_scope(name):
                # NOTE: tf.data.Dataset.list_files shuffles records_glob
                records = tf.data.Dataset.list_files(file_patterns)
                dataset = tf.data.TFRecordDataset(records)

                if epochs != 0:
                    dataset = dataset.repeat(epochs)

                # Parse TFRecord
                parse_fn = lambda x: tf.io.parse_single_example(
                                     x, tf_fmt, name="ParseExample")
                dataset = dataset.map(parse_fn,
                                      num_parallel_calls=_CPU_COUNT-1)
                # Decode images
                if decode_fn == None:
                    if augment == None:
                        decoder = lambda x: \
                                  self._default_decoder(x, crop_and_split=True)
                        dataset = dataset.map(decoder,
                                              num_parallel_calls=_CPU_COUNT-1)
                    else:
                        dataset = dataset.map(self._default_decoder,
                                              num_parallel_calls=_CPU_COUNT-1)
                else:
                    dataset = dataset.map(decode_fn,
                                          num_parallel_calls=_CPU_COUNT-1)
                # Data augmentation
                if augment != None:
                    if callable(augment):
                        dataset = dataset.map(augment,
                                              num_parallel_calls=_CPU_COUNT-1)
                    else:
                        dataset = dataset.map(self._default_augmentation,
                                              num_parallel_calls=_CPU_COUNT-1)

                # Batch and prefetch image/label data
                if self.batch_size > 1:
                    dataset = dataset.batch(self.batch_size)
                dataset = dataset.prefetch(self.batch_size)
                self.datasets[name] = {}
                # Store dataset
                self.datasets[name]["dataset"]  = dataset
                # Store number of ticks in the iterator
                self.datasets[name]["count"]    = \
                    ((len(filenames) - 1) // self.batch_size) + 1
                # Create [re-]initializable iterator
                self.datasets[name]["iterator"] = \
                    dataset.make_initializable_iterator()
                # Add initialization op for this datset
                self.datasets[name]["init"] = \
                    self.datasets[name]["iterator"].initializer
            # END scope @name
        # END scope "Dataset"

        return self.datasets[name]["count"]

    def get_datset(self, name):
        """
        Convenience function to retrieve the dataset
        :param name: dataset identifier specified when @add_dataset was
                     called
        :returns:    the datset
        :rtype:      tf.data.Dataset
        """
        return self.datasets[name]["dataset"]

    def init_iterator(self, name, sess):
        """
        Convenience function to initialize the iterator for the dataset
        referenced by @name.
        NOTE: iterating past the dataset raises tf.errors.OutOfRangeError
        :param name: dataset identifier specified when @add_dataset was
                     called
        :param sess: an active tf.Session to run the initializer
        """
        # Fetch the iterator
        init_op = self.datasets[name]["init"]
        # Run the initializer
        sess.run(init_op)

    def get_output(self, name):
        """
        NOTE: need to run initializer
        """
        return self.datasets[name]["iterator"].get_next()

    def _default_decoder(self, example, crop_and_split=False):
        """
        Decodes all images in example and stacks all channels along with
        the decoded label image in the last channel.

        :param example: parsed example (tf.Example)
        :returns: Stacked tensor with label image in last channel
        :rtype:   tf.Tensor
        """
        images = []
        channels = 0
        # Iterate over example keys and collect all images not label
        for key in example:
            if key.endswith("/data"):
                scope = "/".join(key.split("/")[:-1])
                with tf.name_scope(scope):
                    image = tf.image.decode_image(example[key],
                                                  dtype=tf.uint8,
                                                  name="DecodeImage")
                images.append(image)
                channels += self.input_depths[scope]
        self.shape[-1] = channels
        # Decode label image
        label = tf.image.decode_png(example["label"], channels=1,
                                    name="DecodeLabel")
        # Stack the images' channels and label image
        image_stack = tf.concat(images+[label], axis=2, name="StackImages")
        # Need to recover channel dimension
        stack_shape = image_stack.shape.as_list()
        stack_shape[-1] = self.shape[-1]+1
        image_stack.set_shape(stack_shape)

        if not crop_and_split:
            return image_stack
        else:
            # TODO: move this to a separate function "_default_no_augmentation"
            # Height and width need not be the same accross samples
            stack_shape = tf.shape(image_stack)
            center = [stack_shape[0]//2, stack_shape[1]//2]
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
            return image, label, mask

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
            px_scaling = tf.random.uniform(shape=[channels], maxval=1.5,
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
            image = tf.multiply(image, px_scaling, name="RandomPixelScaling")
            image = tf.clip_by_value(image, 0.0, 1.0,
                                     name="RandomPixelScaling/Clip")
            # Generate mask
            label, mask = generate_mask(label)
        return image, label, mask

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
