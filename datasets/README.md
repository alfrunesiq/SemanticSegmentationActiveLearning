# dataset/support
This directory contains support functions for the specific datasets.

The support layer should provide functions to associate dataset ground-truth
labels with the raw image files, as well as map the label image to an
appropriate set of predefined labels.

## Cityscapes
The labels correspond to the suggested `trainIds` in cityscapes upstream
[repository](https://github.com/mcordts/cityscapesScripts)
> Labels:
| Semantic label       | Label ID | Training ID |
| -------------------- |---------:|------------:|
| Unlabeled            |        0 |         255 |
| Ego vehicle          |        1 |         255 |
| Rectification border |        2 |         255 |
| Out of roi           |        3 |         255 |
| Static               |        4 |         255 |
| Dynamic              |        5 |         255 |
| Ground               |        6 |         255 |
| Road                 |        7 |           0 |
| Sidewalk             |        8 |           1 |
| Parking              |        9 |         255 |
| Rail track           |       10 |         255 |
| Building             |       11 |           2 |
| Wall                 |       12 |           3 |
| Fence                |       13 |           4 |
| Guard rail           |       14 |         255 |
| Bridge               |       15 |         255 |
| Tunnel               |       16 |         255 |
| Pole                 |       17 |           5 |
| Polegroup            |       18 |         255 |
| Traffic light        |       19 |           6 |
| Traffic sign         |       20 |           7 |
| Vegetation           |       21 |           8 |
| Terrain              |       22 |           9 |
| Sky                  |       23 |          10 |
| Person               |       24 |          11 |
| Rider                |       25 |          12 |
| Car                  |       26 |          13 |
| Truck                |       27 |          14 |
| Bus                  |       28 |          15 |
| Caravan              |       29 |         255 |
| Trailer              |       30 |         255 |
| Train                |       31 |          16 |
| Motorcycle           |       32 |          17 |
| Bicycle              |       33 |          18 |
| License plate        |       -1 |         255 |
