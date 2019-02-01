import os

import tensorflow as tf
__all__ = ["InputStage"]

class InputStage:
    def __init__(self, path, fmt, batch_size,
                 size=[512,512],
                 decode_augment_fn=None):
        # TODO insert docstring
        filenames = os.path.join(path, "*.tfrecord")
        self.fmt  = fmt
        self.size = size
        with tf.name_scope("InputStage"):
            # NOTE: tf.data.Dataset.list_files returns a ShuffleDataset
            self.filenames = tf.data.Dataset.list_files(filenames)
            self.dataset   = tf.data.TFRecordDataset(self.filenames)

            parse_fn = lambda x: tf.io.parse_single_example(x, fmt,
                                                            name="ParseExample")
            # Parse TFRecord
            self.dataset = self.dataset.map(parse_fn)
            # Decode and augment
            if decode_augment_fn == None:
                self.dataset = self.dataset.map(self._default_decoder)
                self.dataset = self.dataset.map(self._default_augmentation)
            else:
                self.dataset = self.dataset.map(decode_fn)
            # Batch and prefetch image/label data
            self.dataset = self.dataset.batch(batch_size)
            self.dataset = self.dataset.prefetch(batch_size)

    def get_datset(self):
        return self.dataset

    def _default_decoder(self, example):
        """
        Decodes all images in example and stacks all channels along with the
        decoded label image in the last channel.

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
        print(images)
        # Decode label image
        label = tf.image.decode_png(example["label"], channels=1,
                                    name="DecodeLabel")
        # Stack the images' channels and label image
        image_stack = tf.concat(images+[label], axis=-1, name="StackImages")
        return image_stack

    def _default_augmentation(self, images):
        """
        Applies random crop, flip and channel scaling and separates
        image from labels.
        :param images: image channels and labels stacked
        :returns: images   , label
        :rtype:   tf.Tensor, tf.Tensor
        """
        with tf.name_scope("DataAugmentation"):
            images_shape = tf.shape(images, name="ImagesShape")
            channels     = images_shape[-1]
            crop_shape   = tf.stack(self.size + [channels])
            img_channels = channels - 1
            # Random channelwise pixel intensity scaling
            px_scaling = tf.random.uniform(shape=[img_channels], maxval=1.5,
                                             minval=0.8, dtype=tf.float32)
            # Crop out a sizable portion randomly
            image_crop = tf.random_crop(images, crop_shape,
                                        name="RandomCrop")
            image_crop = tf.image.random_flip_left_right(image_crop)
            # Slice out the image stack and label
            image = image_crop[:,:,:img_channels] # [h,w,c]
            label = image_crop[:,:,img_channels:] # [h,w,1]
            # NOTE this also scales values to range [0.0, 1.0]
            image = tf.image.convert_image_dtype(image, tf.float32,
                                                 name="ToFloat")

            # Apply random channel scaling
            image = tf.multiply(image, px_scaling, name="RandomPixelScaling")
            image = tf.clip_by_value(image, 0.0, 1.0,
                                     name="RandomPixelScaling/Clip")
        return image, label

