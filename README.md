# master-src
File Structure:  
```
.
├── conf/               - Configuration for `train.py` and `active_learning.py`
├── datasets/           - Dataset-specific support layer.
├── deprecated/         - Deprecated code.
├── models/             - Model implementations.
├── tensortools/        - Tensorflow utility library.
├── tools/              - Various tools.
├── util/               - Miscellaneous utility stuff.
├── active_learning.py  - Active learning training script.
├── generate_dataset.py - Dataset preparation script gathering examples to TFRecords.
├── inference.py        - Inference script; run network output to directory.
└── train.py            - Training script.
```

## Dataset TFRecord-format
```python
{
    "image/data"     : tf.io.FixedLenFeature((), tf.string, ""),
    "image/encoding" : tf.io.FixedLenFeature((), tf.string, ""),
    "image/channels" : tf.io.FixedLenFeature((), tf.int64 , -1),
    "label"          : tf.io.FixedLenFeature((), tf.string, ""),
    "height"         : tf.io.FixedLenFeature((), tf.int64 , -1),
    "width"          : tf.io.FixedLenFeature((), tf.int64 , -1),
    "id"             : tf.io.FixedLenFeature((), tf.string, "")
}
```

| Key                   | Description                         |
| --------------------- | ----------------------------------- |
| `image/data`          | encoded (png/jpg) image data.       |
| `image/encoding`      | encoding type (png/jpg/tif).        |
| `image/channels`      | channel count for image.            |
| `label`               | segmentation label mask.            |
| `height`              | image height.                       |
| `width`               | image width.                        |
| `id`                  | unique example id from file prefix. |
| `<modality>/data`     | (optional) same as for `image`      |
| `<modality>/encoding` | (optional) same as for `image`      |
| `<modality>/channels` | (optional) same as for `image`      |
