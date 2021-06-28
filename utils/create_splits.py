import os

# Getting the current work directory (cwd)
labels_path ="D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/results_refined"
destination_path = "D:/Ian/UNI/5_Master_CV/M9_TFM/media/openpcdet/ImageSets"
# r=root, d=directories, f = files
i= 1
for r, d, f in os.walk(labels_path):
    for file in f:
        if file.endswith(".txt"):
            if i%4==0:
                with open(destination_path+'/val.txt', 'a+') as fout:
                    fout.write(file.split(".")[0]+'\n')
            else:
                with open(destination_path+'/train.txt', 'a+') as fout:
                    fout.write(file.split(".")[0]+'\n')
            i+=1