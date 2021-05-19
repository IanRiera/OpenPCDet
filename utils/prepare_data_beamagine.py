import os, argparse
import numpy as np
from plyfile import PlyData

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--all', action='store_true', default=False, help='prepare all scenes')
    # parser.add_argument('--input', type=str, default='/home/oscar/media/frustum/object/training/velodyne/001047.ply')
    # parser.add_argument('--output', type=str, default='/home/oscar/media/frustum/object/training/001047.bin')

    args = parser.parse_args()
    return args

def read_ply(ply_path):
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0].data
    points = np.asarray(vertex.tolist())
    return points

if __name__ == '__main__':
    args = parse_args()

    if args.all:
        input_path = '/mnt/gpid08/users/ian.riera/media/pointcloud/ply'
        output_path = '/mnt/gpid08/users/ian.riera/media/openpcdet/training/velodyne'
        image_sets_path = '/mnt/gpid08/users/ian.riera/media/openpcdet/imageSets/train.txt'

        for subdir, dirs, files in os.walk(input_path):
            for file in files:
                points = read_ply(os.path.join(subdir, file))

                points[:, :3] /= 1000
                points[:, 3] -= np.min(points[:, 3])
                points[:, 3] /= np.max(points[:, 3])

                rot_x = -np.pi / 2
                Rx = np.array([[1, 0, 0],
                               [0, np.cos(rot_x), -np.sin(rot_x)],
                               [0, np.sin(rot_x), np.cos(rot_x)]])

                rot_z = -np.pi / 2
                Rz = np.array([[np.cos(rot_z), -np.sin(rot_z), 0],
                               [np.sin(rot_z), np.cos(rot_z), 0],
                               [0, 0, 1]])

                R = Rz @ Rx

                # add a -1.1 in that position if you want to lower the height of the point cloud (z coordinate), not
                # necessary
                # H = np.row_stack((np.column_stack((R, [0, 0, -1.1])), [0, 0, 0, 1]))

                H = np.row_stack((np.column_stack((R, [0, 0, 0])), [0, 0, 0, 1]))

                for idx, point in enumerate(points):
                    new_point = H @ np.array([*point[:3], 1])
                    new_point = new_point[:3] / new_point[3]
                    points[idx][:3] = new_point

                filename = file.split('.ply')[0]
                with open(image_sets_path, 'a') as f:
                    f.write(filename+'\n')

                points.astype(np.float32).tofile(os.path.join(output_path, filename+'.bin'))