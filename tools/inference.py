import os
import errno 
import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V

import pickle
from datetime import datetime


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        # possibly handle other errno cases here, otherwise finally:
        else:
            raise

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info(torch.cuda.get_device_name(0))
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    data_points = {}
    predictions = {}

    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            annos = demo_dataset.generate_prediction_dicts(
                data_dict, pred_dicts, demo_dataset.class_names,
                output_path='./results/'
            )

            predictions[str(demo_dataset.sample_file_list[idx])] = {
                'boxes': pred_dicts[0]['pred_boxes'].cpu().numpy(),
                'scores': pred_dicts[0]['pred_scores'].cpu().numpy(),
                'labels': pred_dicts[0]['pred_labels'].cpu().numpy()
            }

            data_points[demo_dataset.sample_file_list[idx]] = data_dict['points'][:, 1:].cpu().numpy()

            #V.draw_scenes(
            #    points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            #)
            #mlab.show(stop=True)
    #print(predictions)
    
        
    # datetime object containing current date and time
    now = datetime.now()
    # YYmmdd_HM
    dt_string = now.strftime("%Y%m%d_%H%M")
    print(dt_string)
    results_path = '/mnt/gpid08/users/ian.riera/media/results/{}/'.format(dt_string)
    mkdir_p(results_path)

        
    with open( results_path + 'pred_' + args.cfg_file.split('/')[-1].split('.')[0] + '.pkl', 'wb') as f: #args.data_path.split('/')[-1].split('.')[0] +
        pickle.dump(predictions, f)

    # with open('data_points.pkl', 'wb') as f:
    #     pickle.dump(data_points, f)
    
#---pickle unpackaging---
# results in openpcdet are saved all together in a single pickle. This part divides the global pickle into a single pickle per scene
   
    logger.info('Data postprocessing: single pickle to pickle per scene')
    all_pickle_path = results_path + 'pred_{}.pkl'.format(args.cfg_file.split('/')[-1].split('.')[0])
    pickles_path= results_path+ 'pickles/'
    mkdir_p(pickles_path)
    objects = []


    with (open(all_pickle_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    for key, value in objects[0].items():
        dict = {}
        name = key.split('/')[-1]
        dict[name] = value
        with open(pickles_path + name.split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(dict, f)
    
    logger.info('Pickle extraction done.')

#---pickle to kitti label format (txt)---
# to use the evaluator and refine the results on the lebeller, we need to save the results in txt label format
    logger.info('Data postprocessing: pickle to kitti label format')    
    txt_path = results_path+'txt/'
    mkdir_p(txt_path)

    for filename in os.listdir(pickles_path):
        if filename.endswith(".pkl"):
            objects = []
            with (open(os.path.join(pickles_path,filename), "rb")) as openfile:
                while True:
                    try:
                        objects.append(pickle.load(openfile))
                    except EOFError:
                        break

            for key, value in objects[0].items():
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 h w l x y z heading score 
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 dz dx dy y1 z1 x1 heading score 
                # [x, y, z, dx, dy, dz, heading]

                for i in range(0,len(value["boxes"])):
                    box = value["boxes"][i]
                    f = open(txt_path+"{}.txt".format(filename.split('.')[0]), "a+")
                    f.write("pedestrian -1 -1 -10 -1 -1 -1 -1 {} {} {} {} {} {} {} {}\n".format(box[5],box[3],box[4],box[1],box[2],box[0],box[6], value["scores"][i]))
                    f.close()
    
    logger.info('Txt conversion done')    
    


    logger.info('Inference done.')


if __name__ == '__main__':
    main()
