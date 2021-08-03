import os
from shutil import copyfile

# Getting the current work directory (cwd)
frames_path ="D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/results_refined"
bin_path="D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/velodyne/"
destination_path = "D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/ImageSets"
test_path="D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/test/"
train_path= "D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/train/"
# r=root, d=directories, f = files
i= 1
train_set = ['scene1', 'scene2', 'scene3', 'scene4', 'scene5','scene7', 'scene8']
test_set= ['scene10','scene6', 'scene9']
for r, d, f in os.walk(frames_path):
    for scene in d:
        if scene in train_set:
            for file in os.listdir(r+"/"+scene):
                with open(destination_path+'/train.txt', 'a+') as fout:
                    fout.write(file.split(".")[0]+'\n')
                    copyfile(bin_path+"/"+file.split(".")[0]+".bin", train_path+file.split(".")[0]+".bin")
        elif scene in test_set:
            for file in os.listdir(r+"/"+scene):
                with open(destination_path+'/val.txt', 'a+') as fout:
                    fout.write(file.split(".")[0]+'\n')
                    copyfile(bin_path+"/"+file.split(".")[0]+".bin", test_path+file.split(".")[0]+".bin")
    #for file in f:
    #    if file.endswith(".txt"):
    #        if i%4==0:
    #            with open(destination_path+'/val.txt', 'a+') as fout:
    #                fout.write(file.split(".")[0]+'\n')
    #                copyfile(bin_path+"/"+file.split(".")[0]+".bin", test_path+file.split(".")[0]+".bin")
    #        else:
    #            with open(destination_path+'/train.txt', 'a+') as fout:
    #                fout.write(file.split(".")[0]+'\n')
    #                copyfile(bin_path+"/"+file.split(".")[0]+".bin", train_path+file.split(".")[0]+".bin")
    #        i+=1