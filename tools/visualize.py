import argparse
import glob
from pathlib import Path
from plyfile import PlyData

import mayavi.mlab as mlab
import numpy as np

import sys
sys.path.insert(1, 'D:\\Ian\\UNI\\5_Master_CV\\M9_TFM\\repo')

from OpenPCDet.pcdet.config import cfg, cfg_from_yaml_file
from OpenPCDet.pcdet.utils import common_utils
from OpenPCDet.tools.visual_utils import visualize_utils as V

from functools import partial

import pickle
from copy import deepcopy

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='pointrcnn', help='model used to predict the boxes')
    parser.add_argument('--score', type=float, default=0.6, help='confidence threshold to show detections')
    parser.add_argument('--ped', action='store_true', help='only show pedestrian detections')
    parser.add_argument('--points_path',  type=str, default='D:\\Ian\\UNI\\5_Master_CV\\M9_TFM\\media\\openpcdet\\training\\velodyne\\')#'D:\\Ian\\UNI\\5_Master_CV\\M9_TFM\\media\\kitti\\training\\velodyne\\000008.bin')
    parser.add_argument('--detection_path', type=str, default='D:\\Ian\\UNI\\5_Master_CV\\M9_TFM\\media\\beamagine\\dataset\\results\\pickles\\')#'D:\\Ian\\UNI\\5_Master_CV\\M9_TFM\\results\\openpcdet\\0_demo\\000008_pred_pv_rcnn.pkl')
    parser.add_argument('--results',type=float,default=True,help='show bboxes')
    parser.add_argument('--sample',type=str,default='20200110_135503_976',help='sample name')
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
            if not args.ped or args.ped and p['labels'][idx] == 2:
                if score >= args.score:
                    filtered_boxes.append(p['boxes'][idx])
                    filtered_scores.append(p['scores'][idx])
                    filtered_labels.append(p['labels'][idx])

                    if p['labels'][idx] == 2:
                        print(p['boxes'][idx])
    if args.results:
        V.draw_scenes(
            points=points, ref_boxes=np.array(filtered_boxes),
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