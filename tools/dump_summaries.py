import argparse
import csv
import os
import re
import struct
import sys

import numpy as np
import tensorflow as tf

def main(args):
    scalars_dict = {}
    tensor_dict  = {}
    event = tf.Event()

    with open(args.event_file, "rb") as f:
        # Event file record layout:
        # 8Bytes length
        # 4Bytes CRC-32C of length
        # ...Bytes serialized event
        # 4Bytes CRC-32C of event
        while True:
            header = f.read(12)
            if header == b"":
                break
            length = struct.unpack("=Q", header[:8])[0]
            serialized_event = f.read(length)
            footer = f.read(4)
            event.ParseFromString(serialized_event)
            summary = getattr(event, "summary", None)
            if summary == None:
                # No summary in event
                continue
            step = event.step

            for val in summary.value:
                tag = val.tag
                if re.search(args.summaries, tag) == None:
                    continue
                _type = val.WhichOneof("value")
                if _type == "simple_value":
                    scalars_dict.setdefault(tag, {})[step] = val.simple_value
                elif _type == "tensor":
                    tensor = val.tensor
                    shape  = [dim.size for dim in tensor.tensor_shape.dim]
                    dtype  = tf.dtypes.DType(tensor.dtype)
                    tensor_val = getattr(tensor, dtype.name + "_val")
                    if dtype == tf.string:
                        dtype = np.int64
                    arr    = np.zeros(shape, dtype=dtype)
                    assert np.prod(shape) == len(tensor_val), "Shape mismatch!"
                    for i in range(len(tensor_val)):
                        # Compute shape index from flat index
                        idx = []
                        i_ = i
                        for j in reversed(range(len(shape))):
                            idx.insert(0, i_ % shape[j])
                            i_ = i_ // shape[j]
                        arr[tuple(idx)] = dtype(tensor_val[i])
                    tensor_dict.setdefault(tag, {})[step] = arr
                else:
                    continue

    # Save tensors
    for key, value in tensor_dict.items():
        filename = "_".join(key.split("/")) + ".npz"
        pathname = os.path.join(args.out_dir, filename)
        stacked_array = np.stack(list(value.values()), axis=0)
        with open(pathname, "wb") as f:
            np.savez(f,
                     Step =np.array(list(value.keys())  , dtype=np.int32),
                     Value=np.stack(list(value.values()), axis=0))
    # Save scalars
    for key, value in scalars_dict.items():
        filename = "_".join(key.split("/")) + ".csv"
        pathname = os.path.join(args.out_dir, filename)
        with open(pathname, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Step","Value"], lineterminator="\n")
            writer.writeheader()
            for step, val in value.items():
                _val = val if args.scale_factor == None \
                       else float(val) * args.scale_factor
                writer.writerow({"Step": step, "Value": _val})

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--event-file",
        required=True,
        type=str,
        help="Path to event file"
    )
    parser.add_argument(
        "-o", "--out-dir",
        required=True,
        type=str,
        help="Path to output directory"
    )
    parser.add_argument(
        "-s", "--summaries",
        required=True,
        type=str,
        help="Regex pattern of summaries to include."
    )
    parser.add_argument(
        "--scale-factor",
        required=False,
        default=None,
        type=float,
        help="Scalar summary scale"
    )
    args = parser.parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    sys.exit(main(args))
