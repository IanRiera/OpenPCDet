import os
from shutil import copyfile
# Getting the current work directory (cwd)
labels_path ="D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/label_2"
bin_path="D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/velodyne/"
destination_path = "D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/ImageSets"
test_path="D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/train/"
# r=root, d=directories, f = files
i= 1
for r, d, f in os.walk(labels_path):
    for file in f:
        if file.endswith(".txt"):
            if i%4==0:
                with open(destination_path+'/val_sanity.txt', 'a+') as fout:
                    fout.write(file.split(".")[0]+'\n')
                    
            else:
                with open(destination_path+'/train_sanity.txt', 'a+') as fout:
                    fout.write(file.split(".")[0]+'\n')
                    copyfile(bin_path+"/"+file.split(".")[0]+".bin", test_path+file.split(".")[0]+".bin")
            i+=1