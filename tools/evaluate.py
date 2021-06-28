from eval_utils import kitti_common as kitti
from eval_utils import eval

det_path = "/mnt/gpid08/users/ian.riera/media/results/20210628_1018/txt/"
dt_annos = kitti.get_label_annos(det_path)

gt_path = "/mnt/gpid08/users/ian.riera/media/openpcdet/training/label_2/"
#gt_split_file = "/path/to/val.txt" # from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz
#val_image_ids = _read_imageset_file(gt_split_file)
gt_annos = kitti.get_label_annos(gt_path)
print(eval.get_official_eval_result(gt_annos, dt_annos, 0)) # 6s in my computer
#print(get_coco_eval_result(gt_annos, dt_annos, 0)) # 18s in my computer
