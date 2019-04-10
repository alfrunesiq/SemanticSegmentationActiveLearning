import argparse
import glob
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import numpy as np
import cv2

def parse_label_entry(label):
    entry = {}
    for node in label:
        if node.tag == "name":
            entry["name"] = node.text
        elif node.tag == "attributes":
            for attr in node:
                tag, value = attr.text.split("=")[-1].split(":")
                entry[tag] = value
    return entry

def main(args):
    for filepattern in args.paths:
        if not filepattern.endswith(".xml"):
            continue
        for path in glob.glob(filepattern):
            print(path)
            # Parse xml file
            tree = ET.ElementTree(file=path)
            root = tree.getroot()
            cmap = {}
            labels = None
            # Find labels subnode: root->meta->labels
            for node in root:
                # Find "meta subnode"
                if node.tag == "meta":
                    for sub_node in node:
                        if sub_node.tag == "task":
                            for sub_sub_node in sub_node:
                                if sub_sub_node.tag == "labels":
                                    labels = sub_sub_node
            if labels == None:
                raise ValueError("Could not find labels entry")

            for label in labels:
                attrs = parse_label_entry(label)
                for name in attrs:
                    cmap[attrs["name"]] = int(attrs["id"])
            # TODO Create label map (maybe use metadata field?)

            # Go over all image entries and extract polygons
            for node in root:
                if node.tag != "image":
                    continue
                width  = int(float(node.attrib["width"]))
                height = int(float(node.attrib["height"]))
                name   = node.attrib["name"].split(".")[0]
                annotation = np.full((height, width), cmap["void"], dtype=np.uint8)
                for polygon in node:
                    pts_str = [pt.split(",") \
                               for pt in polygon.attrib["points"].split(";")]
                    # Need to first cast to float before converting to int32
                    # because of decimal string.
                    pts = np.round(np.array(pts_str, np.float32)).astype(np.int32)
                    annotation = cv2.fillPoly(annotation, [pts],
                                              color=cmap[polygon.attrib["label"]]) # NOTE colormap here
                cv2.imwrite(name + "_GT.png", annotation)
                print("written: %s" % name)
                #NOTE imsave here


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o",
                dest="output",
                type=str,
                help="Output directory"
    )
    parser.add_argument(
                nargs=argparse.REMAINDER,
                dest="paths",
                type=str,
                help="Glob paths to xml with polygons"
    )
    args = parser.parse_args()
    main(args)
