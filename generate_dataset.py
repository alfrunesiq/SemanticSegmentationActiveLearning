import argparse
import multiprocessing
import os
import sys

_NUM_CPUS = multiprocessing.cpu_count()

from tensorflow.compat import v1 as tf
tf.disable_eager_execution()

import tqdm

import datasets

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    _bytes = value if not (isinstance(value, str) and sys.version_info[0] == 3) \
                   else value.encode()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[_bytes]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

"""
generator:
   | |
key| |val: path
   | |
write_serialized_example
   |  |
   |  |
key|  |val: encoding
   |  |
   |  |
"""
def generator_from_file_associations(file_associations):
    data_types = file_associations[next(file_associations.__iter__())]
    def generator():
        for file_id in file_associations:
            # Returned entry: [(file_id, ""), (data_type, file_path), ...]
            keys = ["file_id"]
            vals = [file_id]
            for data_type in data_types:
                keys.append(data_type)
                vals.append(file_associations[file_id][data_type])
            yield keys, vals
    return generator, len(data_types)+1

def read_images(keys, paths, dataset, width=None):

    # Noop constants
    _no_shape = tf.constant([0,0,0], dtype=tf.int32)
    _no_data  = tf.constant("", dtype=tf.string)
    def decode_encode_data(arg):
        # Unpack arg
        key, path = arg

        def _center_crop(image, image_shape, aspect_ratio,
                         aspect_old, print_warning):
            # Calculate width
            width = tf.floor(aspect_ratio * tf.cast(image_shape[0],
                                                    tf.float64))
            width = tf.cast(width, tf.int32)
            # Define slice start and endpoints
            top_left = tf.stack([0, (image_shape[1] - width)//2, 0])
            shape    = tf.stack([image_shape[0], width, -1])
            if print_warning:
                # Print debug info
                print_op = tf.print(
                    tf.strings.format("Exmaple: {} with aspect ratio {} is "
                        "center croped to aspect {}\nImage size: {} -> {}\n",
                        (paths[0], aspect_old,
                         width / image_shape[0],
                         image_shape[:2], shape[:2])),
                    output_stream=tf.logging.warning)
                # Crop image to slice (and run @print_op)
                with tf.control_dependencies([print_op]):
                    image_crop = tf.slice(image, begin=top_left, size=shape)
            else:
                # Crop image to slice (no warning)
                image_crop = tf.slice(image, begin=top_left, size=shape)
            return image_crop, shape


        def _label_fun():
            # Read image file
            data_raw = tf.io.read_file(path)
            # Decode file contents
            label = tf.image.decode_image(data_raw)
            # Get image shape
            label_shape = tf.shape(label)
            if args.aspect > 0.0:
                # Center crop
                aspect_ratio = label_shape[1] / label_shape[0]
                label, label_shape  = tf.cond(
                        tf.greater(aspect_ratio, args.aspect),
                        true_fn=lambda: _center_crop(label, label_shape,
                                                     args.aspect,
                                                     aspect_ratio, False),
                        false_fn=lambda: (label, label_shape)
                )
            if width > 0:
                # Downscale to satisfy width but keep aspect ratio.
                scale = width / label_shape[1]
                # Scale height (Truediv yields float64)
                height = tf.cast(label_shape[0], tf.float64)
                height = tf.cast(tf.round(height * scale), tf.int32)
                # Create new size
                size = tf.stack([height, width])
                # Resize with nearest neighbor interpolation
                label = tf.expand_dims(label, axis=0)
                label = tf.image.resize_nearest_neighbor(label, size)
                label = label[0]
                label_shape = tf.shape(label)
            # Get embedding and remap labels to training IDs (see dataset)
            embedding = tf.constant(dataset.embedding, dtype=tf.uint8)
            label = tf.gather_nd(embedding, tf.cast(label, tf.int32))
            # Gather squashes last dimension
            label = tf.expand_dims(label, axis=-1)
            # Get encoded label image
            label_encoding = tf.image.encode_png(label)
            return label_encoding, label_shape

        def _image_fun():
            # Read image file
            data_raw = tf.io.read_file(path)
            image_encoding = data_raw
            # Decode file contents
            image = tf.image.decode_image(image_encoding)
            # Get image shape
            image_shape = tf.shape(image)
            if args.aspect > 0.0:
                # Center crop
                aspect_ratio = image_shape[1] / image_shape[0]
                image, image_shape  = tf.cond(
                        tf.greater(aspect_ratio, args.aspect),
                        true_fn=lambda: _center_crop(image, image_shape,
                                                     args.aspect,
                                                     aspect_ratio, True),
                        false_fn=lambda: (image, image_shape)
                )
            if width > 0:
                # Downscale to satisfy width but keep aspect ratio.
                scale = width / image_shape[1]
                # Scale height (Truediv yields float64)
                height = tf.cast(image_shape[0], tf.float64)
                height = tf.cast(tf.round(height * scale), tf.int32)
                # Create new size
                size = tf.stack([height, width])
                # Resize with bilinear interpolation
                image = tf.expand_dims(image, axis=0)
                image = tf.image.resize_bilinear(image, size)
                image = tf.cast(image, tf.uint8)
                image = image[0]
                # Retrieve resized shape
                image_shape = tf.shape(image)
                # Encode image with same type as input image
                ext = tf.strings.split([path], ".").values[-1]
                image_encoding = tf.cond(
                    tf.equal(ext, "png"),
                    true_fn =lambda: tf.image.encode_png(image),
                    false_fn=lambda: tf.image.encode_jpeg(image))
            return image_encoding, image_shape

        def _noop():
            # file_id Noop
            return _no_data, _no_shape
        encoding, shape = tf.case({
            tf.equal(key, "file_id"): _noop,
            tf.equal(key, "label"): _label_fun
            },
            default=_image_fun)
        return encoding, shape
    # Loop over keys and paths
    encodings, shapes = tf.map_fn(decode_encode_data,
                                  (keys, paths),
                                  dtype=(tf.string, tf.int32))
    return keys, paths, encodings, shapes

def write_serialized_example(keys, paths, encodings, shapes, output_dir):
    """
    Create tf.train.Example
    """
    # keys       = ["file_id", data_type[0], ...]
    # paths      = [@file_id , data_path[0], ...]
    # encodings  = [""       , file_cont[0], ...]
    # shapes     = [""       , shape[0]    , ...]
    # output_dir = str {fixed}
    feature = {}
    for i in range(len(shapes)-1):
        if keys[i] == b"file_id" or keys[i+1] == b"file_id":
            continue
        if shapes[i][0] != shapes[i+1][0] \
            or shapes[i][1] != shapes[i+1][1]:
            raise ValueError("Incompattible shapes (%s and %s): (%s, %s)"
                             % (keys[i], keys[i+1], shapes[i], shapes[i+1]))

    feature["height"] = _int64_feature(shapes[1][0])
    feature["width"]  = _int64_feature(shapes[1][1])
    feature["id"]     = _bytes_feature(paths[0])

    for i in range(1, len(keys)):
        if keys[i] == b"label":
            feature[b"label"] = _bytes_feature(encodings[i])
            aspect_ratio = shapes[i][1] / shapes[i][0]
            if aspect_ratio > 2.0:
                tf.logging.error("%s: example aspect ratio: %1.02f"
                                 % (paths[0].decode("ascii"), aspect_ratio))
        else:
            feature[keys[i] + b"/channels"] = \
                    _int64_feature(shapes[i][-1])
            feature[keys[i] + b"/data"] = \
                    _bytes_feature(encodings[i])
            feature[keys[i] + b"/encoding"] = \
                    _bytes_feature(paths[i].decode("ascii").split(".")[-1])

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    filename = paths[0] + b".tfrecord"
    filename = os.path.join(output_dir, filename)
    with tf.io.TFRecordWriter(filename) as fw:
        fw.write(example.SerializeToString())
    return filename

def tf_write_serialized_example(keys, paths, encoding, shapes, output_dir):
    tf_string = tf.py_func(
        write_serialized_example,
        [keys, paths, encoding, shapes, output_dir],
        tf.string
    )
    # return (file_id, serailized example)
    return tf_string

def main(args):
    tf.logging.set_verbosity(tf.logging.ERROR)
    dataset = None
    if args.dataset.lower() == "cityscapes":
        dataset = datasets.Cityscapes(args.use_coarse)
    elif args.dataset.lower() == "freiburg":
        dataset = datasets.Freiburg(args.modalities)
    elif args.dataset.lower() == "vistas":
        dataset = datasets.Vistas()
    elif args.dataset.lower() == "generic":
        dataset = datasets.Generic(args.image_dir, args.label_dir)
    else:
        raise ValueError("Dataset \"%s\" not supported." % args.dataset)

    if not os.path.exists(args.output_dir):
        sys.stdout.write("Directory \"%s\" does not exist. "
                         % args.output_dir)
        sys.stdout.write("Do you want to create it? [y/N] ")
        sys.stdout.flush()
        user_input = sys.stdin.read(1)
        if user_input.lower()[0] != "y":
            sys.exit(0)
        else:
            os.makedirs(args.output_dir)

    file_associations = dataset.file_associations(args.data_dir)
    sess = tf.Session()
    for split in file_associations:
        # Create path to split
        split_path = os.path.join(args.output_dir, split)
        if not os.path.exists(split_path):
            os.mkdir(split_path)

        # Create generator and retrieve the length
        generator, output_len = generator_from_file_associations(file_associations[split])
        # Create dataset from generator
        tf_dataset = tf.data.Dataset.from_generator(generator,
                output_types=(tf.string, tf.string),
                output_shapes=(output_len, output_len))
        # Add fixed arguments @dataset / @split_path to map functions
        _read_images = lambda x, y: read_images(x, y, dataset, args.width)
        _tf_write_serialized_example = lambda x, y, z, u: \
                tf_write_serialized_example(x, y, z, u, split_path)
        # Map the above functions
        tf_dataset = tf_dataset.map(_read_images,
                num_parallel_calls=_NUM_CPUS-1)
        tf_dataset = tf_dataset.map(_tf_write_serialized_example,
                num_parallel_calls=_NUM_CPUS-1)
        tf_dataset = tf_dataset.batch(_NUM_CPUS-1)
        # Create iterator
        _iter = tf_dataset.make_one_shot_iterator()
        _next = _iter.get_next()
        # Run over all examples
        with tqdm.tqdm(total=len(file_associations[split]),
                       ascii=" #",
                       desc="%-6s" % split,
                       dynamic_ncols=True) as pbar:
            while True:
                try:
                    filenames = sess.run(_next)
                    pbar.update(len(filenames))
                except tf.errors.OutOfRangeError:
                    break
    sess.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root",
                        type=str,
                        dest="data_dir",
                        required=True,
                        help="Path to data set root directory.")
    parser.add_argument("-t", "--dataset",
                        type=str,
                        dest="dataset",
                        required=True,
                        help="Name of the dataset "
                             "{cityscapes,freiburg,vistas,generic}.")
    parser.add_argument("-o", "--output_dir",
                        type=str,
                        dest="output_dir",
                        required=True,
                        help="Path to where to store the records.")
    parser.add_argument("-w", "--width",
                        type=int,
                        default=-1,
                        dest="width",
                        required=False,
                        help="Width of packed examples.")
    parser.add_argument("-a", "--max-aspect-ratio",
                        type=float,
                        default=-1.0,
                        dest="aspect",
                        required=False,
                        help="Downscaling factor.")
    parser.add_argument("--use-coarse",
        action="store_true",
        dest="use_coarse",
        default=False,
        help="(Cityscapes) Use coarse annotation set."
    )
    parser.add_argument("--modalities",
        nargs="*",
        type=str,
        dest="modalities",
        default=None,
        help="(Freiburg) list of modalities to use."
    )
    parser.add_argument("-i", "--image-dir",
        type=str,
        dest="image_dir",
        default=None,
        help="(Generic) Image subdirectory under data root for generic dataset"
    )
    parser.add_argument("-l", "--label-dir",
        type=str,
        dest="label_dir",
        default=None,
        help="(Generic) Label subdirectory under data root for generic dataset"
    )
    args = parser.parse_args()
    main(args)
