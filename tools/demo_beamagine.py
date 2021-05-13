import argparse
import glob
from pathlib import Path
from plyfile import PlyData

import mayavi.mlab as mlab
import numpy as np
import torch

import sys
sys.path.append('../')

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V

from functools import partial

import pickle
from copy import deepcopy

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default='pointpillar', help='model used to predict the boxes')
    parser.add_argument('--score', type=float, default=0.0, help='confidence threshold to show detections')
    parser.add_argument('--ped', action='store_true', help='only show pedestrian detections')

    parser.add_argument('--sample', type=str, default='s1_20200110_135907_139')

    parser.add_argument('--kitti', action='store_true')

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


    if args.kitti:

        points = np.fromfile('/home/oscar/media/frustum/object/training/bin/' + args.sample + '.bin',
                             dtype=np.float32).reshape(-1, 4)
        pcdet_path = '/home/oscar/media/frustum/object/training/openpcdet/pretrained_kitti/' + args.sample + '_pred_'
        with open(pcdet_path + args.model + '.pkl',
                  'rb') as f:
            pred = pickle.load(f)
        p = pred['../pointclouds_beamagine/kitti_' + args.sample + '.bin']

    else:
        points = np.fromfile('/home/oscar/media/frustum/object/training/bin/' + args.sample + '.bin', dtype=np.float32).reshape(-1,4)
        pcdet_path = '/home/oscar/media/frustum/object/training/openpcdet/trained_beamagine/' + args.sample + '_pred_'
        with open(pcdet_path + args.model + '.pkl',
                  'rb') as f:
            pred = pickle.load(f)
        p = pred['../pointclouds_beamagine/' + args.sample + '.bin']

    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []

    for idx, score in enumerate(p['scores']):
        if not args.ped or args.ped and p['labels'][idx] == 2:
            if score >= args.score:
                filtered_boxes.append(p['boxes'][idx])
                filtered_scores.append(p['scores'][idx])
                filtered_labels.append(p['labels'][idx])

                if p['labels'][idx] == 2:
                    print(p['boxes'][idx])

    V.draw_scenes(
        points=points, ref_boxes=np.array(filtered_boxes),
        ref_scores=np.array(filtered_scores), ref_labels=np.array(filtered_labels)
    )

    mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()