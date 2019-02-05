import json
import logging
import os

import tensorflow as tf
from google.protobuf.json_format import MessageToJson

__all__ = ["InputStage"]

def _peek_tfrecord(filepath):
    """
    Parses the first example from a tfrecord, and returns a dictionary
    of the first tf.train.Feature
    :param filepath: path to a tfrecord containing examples
    :returns: first feature in the record
    :rtype:   dict

    """
    example = tf.train.Example()
    rec_iter = tf.io.tf_record_iterator(filepath)
    record = next(rec_iter)
    example.ParseFromString(record)
    # Convert to json and parse to dict
    json_msg = MessageToJson(example)
    features = json.loads(json_msg)
    # Strip off redundant keys
    fmt = features["features"]["feature"]
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

        if len(input_shape) == 3:
            self.shape = input_shape
        elif len(input_shape) == 2:
            self.shape = input_shape + [None]
        else:
            logger.warn("Proceeding with unknown inputshape.")
            self.shape = [None,None,None]

        self.batch_size = batch_size
        self.datasets   = {}
        self.iterator   = None

    def add_dataset(self, name, path, epochs=1, augment=None, decode_fn=None):
        """
        Add a new dataset to the collection self.datasets dictionary.

        NOTE: All added datasets are assumed to have the same output-
              types. The addition of multiple datasets are mainly for the
              puropose of containint train-, validation- and testset.

        :param name:      dataset identifier used to reference the data-
                          set for retrieving dataset/iterator
        :param path:      path to the pre-generated *.tfrecord files
        :param augment:   Can either be
                          a callable : this function is mapped after the
                                       @decode_fn
                          None       : leaves the images as is
                          other      : applies default augmentation
        :param decode_fn: Optional custom decode function to decode the
                          parsed TFRecord examples. The function takes a
                          parsed example as argument.
        """
        records_glob = []
        filenames = []
        if not isinstance(path, list):
            path = [path]
        for _dir in path:
            records_glob.append(os.path.join(_dir, "*.tfrecord"))
            filenames.extend(os.listdir(_dir))
        # Peek into a tfrecord to retrieve format and channel counts
        fmt = _peek_tfrecord(os.path.join(path[0],filenames[0]))
        self.input_depths = {}
        # Extract channel info for the input data
        for key in fmt.keys():
            _key = str(key)
            scope = "/".join(_key.split("/")[:-1])
            if _key.endswith("channels"):
                self.input_depths[scope] = int(fmt[key]["int64List"]["value"][0])

        tf_fmt = _example_from_format(fmt)
        parse_fn = lambda x: tf.io.parse_single_example(x, tf_fmt,
                                                        name="ParseExample")
        with tf.name_scope("Dataset"):
            with tf.name_scope(name):
                # NOTE: tf.data.Dataset.list_files shuffles records_glob
                self.records_glob = tf.data.Dataset.list_files(records_glob)
                dataset   = tf.data.TFRecordDataset(self.records_glob)

                if epochs > 1:
                    dataset = dataset.repeat(epochs)

                # Parse TFRecord
                dataset = dataset.map(parse_fn)
                # Decode images
                if decode_fn == None:
                    if augment == None:
                        decoder = lambda x: \
                                  self._default_decoder(x, crop_and_split=True)
                        dataset = dataset.map(decoder)
                    else:
                        dataset = dataset.map(self._default_decoder)
                else:
                    dataset = dataset.map(decode_fn)
                # Data augmentation
                if augment != None:
                    if callable(augment):
                        dataset = dataset.map(augment)
                    else:
                        dataset = dataset.map(self._default_augmentation)

                # Batch and prefetch image/label data
                if self.batch_size > 1:
                    dataset = dataset.batch(self.batch_size)
                dataset = dataset.prefetch(self.batch_size)
            # END scope @name

            self.datasets[name] = {}
            self.datasets[name]["dataset"] = dataset
            self.datasets[name]["count"]   = len(filenames)
            if self.iterator == None:
                self.iterator = tf.data.Iterator.from_structure(
                    output_types=dataset.output_types,
                    output_shapes=dataset.output_shapes,
                    output_classes=dataset.output_classes,
                    shared_name="DatasetIterator")
            self.datasets[name]["init"] = \
                self.iterator.make_initializer(dataset)

        return self.datasets[name]["count"]
        # END scope "Dataset"

    def get_datset(self, name):
        """
        Convenience function to retrieve the dataset
        :param name: dataset identifier specified when @add_dataset was
                     called
        :returns:    the datset
        :rtype:      tf.data.Dataset
        """
        return self.datasets[name]["dataset"]

    def output(self, name, sess):
        """
        Convenience function to [re-] initialize the iterator output and
        get a handle for the output.
        NOTE: iterating past the dataset raises tf.errors.OutOfRangeError
        :param name: dataset identifier specified when @add_dataset was
                     called
        :param sess: an active tf.Session to run the initializer
        :returns: output handle for the dataset iterator
        :rtype:   tf.Tensor
        """
        sess.run(self.datasets[name]["init"])
        return self.iterator.get_next()

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
                                                 self.size[0], self.size[1])
            image = crop[:,:,:self.shape[-1]]
            label = crop[:,:,self.shape[-1]:]
            return image, label

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
        return image, label

