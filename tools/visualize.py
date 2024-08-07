import argparse
import os
import sys
from pathlib import Path
import time

import mayavi.mlab as mlab
from visual_utils import visualize_utils as V

import torch
import numpy as np
import glob

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets import DatasetTemplate

class VisDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, pcd_path=None, cloud_ext='npy',
                 label_path=None, sample_list_path=None, logger=None):
        """
        Args:
            dataset_cfg: Configuration file of the dataset (E.g. custom_dataset.yaml)
            class_names: Names of the classes we want to detect
            root_path: Path to the directory containing the point cloud files
            sample_list: Path to a text file containing the names of the samples to
                         perform inference on.
            ext: Extension of point cloud data files
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=False, root_path=pcd_path, logger=logger
        )
        self.root_path = pcd_path
        self.cloud_ext = cloud_ext

        self.pcd_file_list = self.create_file_list(pcd_path, self.cloud_ext, sample_list_path)
        self.sample_names = [os.path.splitext(os.path.basename(file))[0] for file in self.pcd_file_list]
        
        if label_path:
            self.label_file_list = self.create_file_list(label_path, 'txt', sample_list_path)
            labels_sample_names = [os.path.splitext(os.path.basename(file))[0] for file in self.label_file_list]
            if self.sample_names != labels_sample_names:
                raise argparse.ArgumentError(None, "Point cloud and label samples do not match.")
        else:
            self.label_file_list = []
                         
    def create_file_list(self, path, file_ext, sample_list_path=None):
        file_list = []
        if path.is_dir():
            if sample_list_path:
                sample_names = []
                with open(sample_list_path, 'r') as sample_list:
                    sample_names = sample_list.read().splitlines()
                for sample_name in sample_names:
                    file_path = str(path / f'{sample_name}.{file_ext}')
                    if os.path.exists(file_path):
                        file_list.append(file_path)
                    else:
                        self.logger.info(f"The path {file_path} does not exist.")
            else:
                file_list = glob.glob(str(path / f'*{file_ext}'))
        else:
            if sample_list_path:
                 raise argparse.ArgumentError(None, '''Cannot use a list of samples together with
                                                    a single point cloud or label file.''')
            if os.path.exists(path):
                file_list.append(str(path))
            else:
                self.logger.info(f"The path {path} does not exist.")

        file_list.sort()
        
        return file_list
    
    def __len__(self):
        return len(self.pcd_file_list)

    def read_labels(self, label_file_path):
        gt_boxes = []
        gt_names = []
        
        with open(label_file_path, 'r') as label_file:
            for line in label_file:
                gt_box_info = line.strip().split()
                gt_boxes.append([float(value) for value in gt_box_info[:-1]]) 
                gt_names.append(gt_box_info[-1]) 

            gt_boxes = np.array(gt_boxes)
            gt_names = np.array(gt_names)

        return gt_boxes, gt_names

    def __getitem__(self, index):
        if self.cloud_ext == 'bin':
            points = np.fromfile(self.pcd_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.cloud_ext == 'npy':
            points = np.load(self.pcd_file_list[index])
        else:
            raise NotImplementedError
        
        if self.label_file_list:
            gt_boxes, gt_names = self.read_labels(self.label_file_list[index])    
            input_dict = {
                'points': points,
                'frame_id': index,
                'gt_boxes': gt_boxes,
                'gt_names': gt_names
            }

        else:
            input_dict = {
                'points': points,
                'frame_id': index
            }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='''Python script to perform inference on
                                     specified data and visualize the detections.''')
    parser.add_argument('--cfg_file', type=str, default=None,
                        help='Specify path to the model config for inference.')
    parser.add_argument('--pcd_path', type=str, default=None,
                        help='''Specify path to a single point cloud or a directory
                        containing point clouds. Extension of point clouds is
                        given by 'ext' argument.''')
    parser.add_argument('--weights', type=str, default=None,
                        help='Specify path to the weights you want to use for inference.')
    parser.add_argument('--labels', type=str, default=None,
                        help='''Path to the directory in which the ground truth labels are stored.''')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='''Path to directory in which the visualizations will be saved.
                        If None visualizations will not be saved.''')
    parser.add_argument('--show', action='store_true',
                        help='''Specify whether the visualizations should be displayed.''')
    parser.add_argument('--sample_list', type=str, default=None,
                        help='''Specify path to a text file containing the names of the samples
                        you want to inference on and visualize the predictions. For example, if you
                        want inference on the the point clouds '00008.npy', '00255.npy' and '00021.npy'
                        the text file should contain three lines: \n
                        00008 \n
                        00255 \n
                        00021 \n
                        This can be helpful if you only want to inference on the test or validation data
                        specified in the ImageSets directory. If no path is specified, all samples will be
                        visualized. 'sample_list' argument can only be used if 'pcd_path' is a directory.''')
    parser.add_argument('--ext', type=str, default='npy',
                        help='''Select extension of point cloud data files between 'npy' and 'bin'.
                        Defaults to 'npy'.''')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()

    logger = common_utils.create_logger()
    logger.info('-----------------Visualization-------------------------')

    if args.labels:
        label_path = Path(args.labels)
    else:
        label_path = None

    vis_dataset = VisDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        pcd_path=Path(args.pcd_path), cloud_ext=args.ext, label_path=label_path,
        sample_list_path=args.sample_list, logger=logger
    )

    logger.info(f'Total number of samples: \t{len(vis_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=vis_dataset)
    model.load_params_from_file(filename=args.weights, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx in range(0, len(vis_dataset)):
            data_dict = vis_dataset[idx]
            data_dict = vis_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)

            sample_name = vis_dataset.sample_names[idx]
            logger.info(f'Visualize sample: \t{sample_name}')

            pred_dicts, _ = model(data_dict)
 
            if not label_path:
                gt_boxes = None
            else:
                gt_boxes = data_dict['gt_boxes'][:,:,:-1].squeeze(0)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes=gt_boxes,
                ref_boxes=pred_dicts[0]['pred_boxes'], ref_scores=pred_dicts[0]['pred_scores'],
                ref_labels=pred_dicts[0]['pred_labels'], show=args.show
            )
    
            if args.show:
                mlab.show()
            if args.save_dir:
                image_path = os.path.join(args.save_dir, f"screenshot_{sample_name}.png")
                mlab.savefig(image_path, size=(3840, 2160))

    logger.info('Inference and visualization done.')

if __name__ == '__main__':
        main()