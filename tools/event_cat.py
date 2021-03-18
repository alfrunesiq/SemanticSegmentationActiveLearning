import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input-files",
    nargs="+",
    type=str,
    help="Ordered list of event files to be concatenated")

args = parser.parse_args()

from tensorflow.compat import v1 as tf
import struct

output_filename = args.input_files[0] + ".cat"
event = tf.Event()

with open(output_filename, "wb+") as f_out:
    with open(args.input_files[0], "rb") as f_in:
        f_out.write(f_in.read())

    for filename in args.input_files[1:]:
        print(filename)
        with open(filename, "rb") as f_in:
            header = f_in.read(12)
            length = struct.unpack("=Q", header[:8])[0]
            body = f_in.read(length)
            event.ParseFromString(body)
            print(event)
            footer = f_in.read(4)
            f_out.write(f_in.read())

