# OpenPCDet
This is a fork of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). It adds a visualization tool and configuration files for training on custom datasets.

Original README can be found [here](README.og.md).

## Visualization

The `tools/visualize.py` script runs inference on point cloud data using an OpenPCDet model and visualizes the detected objects.

It takes the following arguments:
- `--cfg_file`: Path to the model config file for inference. **(required)**
- `--weights`: Path to the model weights file for inference. **(required)**
- `--pcd_path`: Path to a single point cloud file or a directory containing multiple point clouds. The file extension is specified by the --ext argument. **(required)**
- `--labels`: Path to the directory containing ground truth labels. **(optional)**
- `--save_dir`: Path to the directory where visualizations will be saved. If not specified, visualizations will not be saved. **(optional)**
- `--show`: Display the visualizations. **(optional)**
- `--sample_list`: Path to a text file listing the names of samples to inference and visualize. This is particularly useful if you only want to inference on test or validation data specified in the `ImageSets` directory. The text file should list one sample name per line. For example, if you want inference on the point clouds `00008.npy`, `00255.npy` and `00021.npy` the text file should contain three lines:
    ```txt
    00008
    00255
    00021
    ```
    If not specified, all samples will be visualized. This argument can only be used if `--pcd_path` is a directory. **(optional)**
- `--ext`: Extension of the point cloud data files (npy or bin). Defaults to npy. **(optional)**

#### Example Usage
```bash
python visualize.py --cfg_file path/to/config.yaml --weights path/to/weights.pth --pcd_path path/to/pointclouds.npy --show
```