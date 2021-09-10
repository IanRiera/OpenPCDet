from shutil import copyfile

txt_path = "/mnt/gpid08/users/ian.riera/media/openpcdet/ImageSets/train_gt.txt"
src = "/mnt/gpid08/users/ian.riera/media/openpcdet/training/velodyne/"
dst = "/mnt/gpid08/users/ian.riera/media/openpcdet/fusion_dataset/"

with open(txt_path) as fp:
  for line in fp:
    copyfile(src+"{}.bin".format(line.strip()), dst+"{}.bin".format(line.strip()))
  