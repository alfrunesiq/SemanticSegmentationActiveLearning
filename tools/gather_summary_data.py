import argparse
import collections
import csv

import tensorflow as tf

def main(args):
    summary_iterator = tf.train.summary_iterator(args.filename)
    events = collections.OrderedDict()

    for event in summary_iterator:
        if not isinstance(event, tf.summary.Event): # Not an event
            continue
        if len(event.ListFields()) < 3: # No step info
            continue

        what = event.ListFields()[2][0].name
        if what.lower() != "summary": # Not a summary
            continue

        summary = event.summary
        for value in summary.value:
            if value.tag in args.names:
                if event.step not in events.keys():
                    events[event.step] = {}
                events[event.step][value.tag] = value.simple_value

    with open(args.output, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["step"] + args.names)
        writer.writeheader()
        for step in events:
            row = events[step]
            row["step"] = step
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        dest="filename",
        required=True,
        help="Path to event file."
    )
    parser.add_argument(
        "-o", "--output-file",
        dest="output",
        required=True,
        help="Path to event file."
    )
    parser.add_argument(
        "-n", "--names",
        nargs=argparse.REMAINDER,
        dest="names",
        required=True,
        help="Name of the summaries to be extracted."
    )
    args = parser.parse_args()
    main(args)
