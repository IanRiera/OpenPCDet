import argparse
import glob
from pathlib import Path
from plyfile import PlyData

import mayavi.mlab as mlab
import numpy as np

import sys
sys.path.append('D:/Ian/UNI/5_Master_CV/M9_TFM/repo/OpenPCDet')

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from tools.visual_utils import visualize_utils as V

from functools import partial

import pickle
from copy import deepcopy

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='pointrcnn', help='model used to predict the boxes')
    parser.add_argument('--score', type=float, default=0.5, help='confidence threshold to show detections')
    parser.add_argument('--ped', action='store_true', help='only show pedestrian detections')
    parser.add_argument('--points_path',  type=str, default='D:/Ian/UNI/5_Master_CV/M9_TFM/media/openpcdet/training/velodyne/')#D:/Ian/UNI/5_Master_CV/M9_TFM/media/openpcdet/training/velodyne/')#'D:/Ian/UNI/5_Master_CV/M9_TFM/media/kitti/training/velodyne/000008.bin')
    parser.add_argument('--detection_path', type=str, default='D:/Ian/UNI/5_Master_CV/M9_TFM/6_results/20210801_225632_40k/pickles/')#D:/Ian/UNI/5_Master_CV/M9_TFM/6_results/20210721_182632/pickles_1/')
    parser.add_argument('--gt_path', type=str, default='D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/pickles/')#'D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/pickles/')
    parser.add_argument('--results',type=float,default=True,help='show bboxes')
    parser.add_argument('--sample',type=str,default='20200219_183356_108',help='sample name')#20200219_184540_543
    args = parser.parse_args()

    return args

def read_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0].data
    points = np.asarray(vertex.tolist())
    return points

def main():
    args = parse_config()
    logger = common_utils.create_logger()

    points = np.fromfile(args.points_path+args.sample+'.bin',
                            dtype=np.float32).reshape(-1, 4)
    if args.results:
        with open(args.detection_path+args.sample+'.pkl',
                    'rb') as f:
            pred = pickle.load(f)
        #p = pred['../pointclouds_beamagine/kitti_' + args.sample + '.bin']


        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []

        p = list(pred.values())[0]
        for idx, score in enumerate(p['scores']):
 
            if score >= args.score:
                filtered_boxes.append(p['boxes'][idx])
                filtered_scores.append(p['scores'][idx])
                filtered_labels.append(p['labels'][idx])

        with open(args.gt_path+args.sample+'.pkl',
                    'rb') as f_gt:
            gt = pickle.load(f_gt)
        #p = pred['../pointclouds_beamagine/kitti_' + args.sample + '.bin']


        gt_boxes = []
        gt_scores = []
        gt_labels = []

        g = list(gt.values())[0]
        for idx, score in enumerate(g['scores']):
            gt_boxes.append(g['boxes'][idx])
            gt_scores.append(g['scores'][idx])
            gt_labels.append(g['labels'][idx])


        #print(g['boxes']) 
    if args.results:
        V.draw_scenes(
            points=points, gt_boxes = np.array(gt_boxes), ref_boxes=np.array(filtered_boxes),
            ref_scores=np.array(filtered_scores), ref_labels=np.array(filtered_labels)
        )
    else:
        V.draw_scenes(
            points=points
        )
    mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()