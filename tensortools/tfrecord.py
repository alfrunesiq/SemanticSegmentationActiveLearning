import struct

from tensorflow.compat import v1 as tf
from google.protobuf.json_format import MessageToDict
from google.protobuf.descriptor import FieldDescriptor

def tfrecord2tfexamples(filename, num_examples=1):
    """
    TFRecord format:
    [
        {
            uint64 : record length
            uint32 : masked crc of length
            bytes  : tf.Example
            uint32 : masked crc of data
        },
        ...
    ]
    We will ignore the crc checks as it uses CRC32C witch isn't natively
    supported in the buildt in python libraries.
    :param filename:     path to tfrecord file
    :param num_examples: number of examples to parse
    :return: list parsed tf.Example(s)
    """
    examples = []
    with open(filename, "rb") as f:
        for i in range(num_examples):
            example  = tf.train.Example()
            record_length = struct.unpack("=Q", f.read(8))[0]
            length_crc    = f.read(4)
            record_bin    = f.read(record_length)
            data_crc      = f.read(4)
            example.ParseFromString(record_bin)
            examples.append(example)
    if num_examples == 1:
        return examples[0]
    else:
        return examples

def tfrecord_iterator(filename):
    def record_iterator():
        with open(filename, "rb") as f:
            while True:
                # header = [record_length{8}, crc32{4}]
                header = f.read(12)
                if header == b"":
                    break
                record_length = struct.unpack("=Q", header[:8])[0]
                event_bin = f.read(record_length)
                footer = f.read(4) # data crc
                yield event_bin
    return record_iterator()

def parse_single_example(filename, fmt):
    # Read raw file contents
    file_content = tf.io.read_file(filename)
    # Decode filst 8 bytes as length
    length_bytes = tf.strings.substr(file_content, 0, 8)
    length = tf.io.decode_raw(length_bytes, tf.int64)
    length_int32 = tf.cast(length[0], tf.int32)
    # Extract serialized record substring
    serialized_record = tf.strings.substr(file_content, 12, length_int32)
    # Parse example to dict
    example = tf.io.parse_single_example(serialized_record, fmt)
    return example

def read_tfrecord(filename):
    # NOTE: skips crc check
    with tf.io.gfile.GFile(filename, "rb") as f:
        header = f.read(12)
        if header == b"":
            return b""
        record_length = struct.unpack("=Q", header[:8])[0]
        serialized_record = f.read(record_length)
    return serialized_record

def tfrecord2example_dict(filename):
    return MessageToDict(tfrecord2tfexamples(filename))

