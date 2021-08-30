import pickle
import os
import numpy as np


if __name__ == '__main__':

    #src_path=  'D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/corrected_labels/'
    #dst_path=  'D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/corrected_prepared/'
    src_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/kitti/training/label_2/'
    dst_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/kitti/training/label_2_prepared/'

    for filename in os.listdir(src_path):
        if filename.endswith(".txt"):
            objects = []
            file1 = open(os.path.join(src_path,filename), "r")
            while True:
                #count += 1
 
                # Get next line from file
                line = file1.readline()
 
                # if line is empty
                # end of file is reached
                if not line:
                    break
                objects.append(line.strip())
 
            file1.close()

            
            for object in objects:
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 dz dy dx -y1 z1 x1 heading score 
                # 0           1  2  3   4  5  6  7  8  9 10  11 12 13   14     
                # [x, y, z, dx, dy, dz, heading]
                # 13 -11 12 10  9  8  14
                tokens = object.split(' ')
                
                f = open(dst_path+"{}.txt".format(filename.split('.')[0]), "a+")
                #from refining to openpcdet
                f.write("Pedestrian -1 -1 -10 -1 -1 -1 -1 {} {} {} {} {} {} {} {}\n".format(tokens[8],tokens[9],tokens[10],tokens[11],-float(tokens[12])+(float(tokens[8])/2),tokens[13],tokens[14], tokens[15]))

                f.close()
 

        else:
            continue

