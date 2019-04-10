from __future__ import print_function, absolute_import, division

import glob
import os

import numpy as np

class Generic:

    def __init__(self, image_dir=None, label_dir=None):
        self.image_dir=image_dir
        self.label_dir=label_dir
        pass

    def file_associations(self, root_path):
        image_dir = os.path.join(root_path, "images") if self.image_dir==None \
                    else os.path.join(root_path, self.image_dir)
        label_dir = os.path.join(root_path, "labels") if self.label_dir==None \
                    else os.path.join(root_path, self.label_dir)
        _file_associations = {"examples" : {}}
        print(image_dir)
        if os.path.exists(image_dir) and os.path.isdir(image_dir):
            for root, dirs, filenames in os.walk(image_dir):
                for filename in filenames:
                    if not (filename.endswith(".png") or 
                            filename.endswith(".jpg")):
                        continue
                    # Strip off extension
                    file_id = ".".join(filename.split(".")[:-1])
                    image_path = os.path.join(root, filename)
                    # Add file-association entry
                    _file_associations["examples"][file_id] = {}
                    _file_associations["examples"][file_id]["image"] = image_path

                    # Construct label_path
                    label_path = os.path.join(label_dir, # label root dir
                                              root.strip(image_dir), # file subdir
                                              file_id+"*") # file glob
                    # Expand glob
                    label_path = glob.glob(label_path)
                    if len(label_path) > 1:
                        raise ValueError(
                                "Every label must have same filename"
                                "as corresponding image.")
                    elif len(label_path) == 0:
                        print("[INFO] Example \"%s\" has no matching label." % file_id)
                    else:
                        label_path = label_path[0]
                        _file_associations["examples"][file_id]["label"] = label_path
                    

        return _file_associations

    @property
    def embedding(self):
        # Identity embedding
        return np.arange(256, dtype=np.uint8)

