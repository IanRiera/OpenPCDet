import pickle
import os
import numpy as np


if __name__ == '__main__':
    #src_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/corrected_labels/'
    #dst_path = 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/corrected_labels/'
    #src_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/kitti/training/label_2/'
    #dst_path = 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/kitti/training/pickles/'
    #train_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/kitti/training/pickles_train/'
    #test_path = 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/kitti/training/pickles_test/'

    src_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/demos/'
    dst_path = 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/demos/'

    for count,filename in enumerate(os.listdir(src_path)):
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

            #objects_pkl = []
            #with (open(aux_path+"{}.pkl".format(filename.split('.')[0]), "rb")) as openfile:
            #    while True:
            #        try:
            #            objects_pkl.append(pickle.load(openfile))
            #        except EOFError:
            #            break     

            name = filename.split('.')[0]+'.bin'
            dict={name: {'boxes':[],'labels':[],'scores':[]}}
            
            for object in objects:
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 dz dy dx -y1 z1 x1 heading score 
                # 0           1  2  3   4  5  6  7  8  9 10  11 12 13   14     
                # [x, y, z, dx, dy, dz, heading]
                # 13 -11 12 10  9  8  14
                tokens = object.split(' ')
                box = np.asarray([float(tokens[13]),-float(tokens[11]),float(tokens[12]),float(tokens[10]),float(tokens[9]),float(tokens[8]),float(tokens[14])])
                dict[name]['boxes'].append(box)
                dict[name]['labels'].append(1)
                dict[name]['scores'].append(float(tokens[-1]))

            #print(dict)
            #if (count+1) % 4 == 0:
            #    with open(test_path + filename.split('.')[0] + '.pkl', 'wb') as f: 
            #        pickle.dump(dict, f)
            #else:
            #    with open(train_path + filename.split('.')[0] + '.pkl', 'wb') as f: 
            #        pickle.dump(dict, f)
 
            with open(dst_path + filename.split('.')[0] + '.pkl', 'wb') as f: 
                pickle.dump(dict, f)

        else:
            continue

