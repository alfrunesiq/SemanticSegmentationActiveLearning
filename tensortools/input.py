import os

import tensorflow as tf

__all__ = ["InputStage"]

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
    def __init__(self, record_fmt, batch_size,
                 input_shape=[512,512,3]):
        """
        Initializes the @InputStage class

        :param record_fmt: dict with TFRecord layout mapping record keys
                           to tf.[Fixed|Var]LenFeature.
        :param batch_size:  Batch size for the dataset pipeline
        :param input_shape: Input shape of the network
        """
        self.fmt        = record_fmt
        self.shape      = input_shape
        self.batch_size = batch_size
        self.datasets   = {}
        self.iterator   = None
        self._init_ops  = {}

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
        filenames = []
        if isinstance(path, list):
            for _dir in path:
                filenames.append(os.path.join(_dir, "*.tfrecord"))
        else:
            filenames = os.path.join(path, "*.tfrecord")
        parse_fn = lambda x: tf.io.parse_single_example(x, self.fmt,
                                                        name="ParseExample")
        with tf.name_scope("Dataset"):
            with tf.name_scope(name):
                # NOTE: tf.data.Dataset.list_files shuffles filenames
                self.filenames = tf.data.Dataset.list_files(filenames)
                dataset   = tf.data.TFRecordDataset(self.filenames)

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

            self.datasets[name] = dataset
            if self.iterator == None:
                self.iterator = tf.data.Iterator.from_structure(
                    output_types=dataset.output_types,
                    output_shapes=dataset.output_shapes,
                    output_classes=dataset.output_classes,
                    shared_name="DatasetIterator")
            self._init_ops[name] = self.iterator.make_initializer(dataset)
        # END scope "Dataset"

    def get_datset(self, name):
        """
        Convenience function to retrieve the dataset
        :param name: dataset identifier specified when @add_dataset was
                     called
        :returns:    the datset
        :rtype:      tf.data.Dataset
        """
        return self.datasets[name]

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
        sess.run(self._init_ops[name])
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
        # Iterate over example keys and collect all images not label
        for key in self.fmt.keys():
            if key.endswith("/data"):
                _type = key.split("/")[0]
                with tf.name_scope(_type):
                    image = tf.image.decode_image(example[key],
                                                  dtype=tf.uint8,
                                                  name="DecodeImage")
                images.append(image)
        # Decode label image
        label = tf.image.decode_png(example["label"], channels=1,
                                    name="DecodeLabel")
        # Stack the images' channels and label image
        image_stack = tf.concat(images+[label], axis=2, name="StackImages")
        image_stack.set_shape([None,None,3])
        if not crop_and_split:
            return image_stack
        else:
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

