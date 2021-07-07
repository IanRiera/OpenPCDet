import pickle
import os
import numpy as np


if __name__ == '__main__':
    src_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/openpcdet_old/training/label_2/'
    dst_path = 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/openpcdet_old/training/pickles/'
    #aux_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/media/beamagine/dataset/results/pickles/'

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
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 h w l x y z heading score 
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 dz dx dy y1 z1 x1 heading score 
                # [x, y, z, dx, dy, dz, heading]
                tokens = object.split(' ')
                #box = np.asarray([float(tokens[13]),float(tokens[11]),float(tokens[12]),float(tokens[9]),float(tokens[10]),float(tokens[8]),float(tokens[14])])
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 dz dy dx y1 -z1 x1 heading score 
                box = np.asarray([float(tokens[13]),-float(tokens[11]),-float(tokens[12]),float(tokens[9]),float(tokens[8]),float(tokens[10]),float(tokens[14])])
                dict[name]['boxes'].append(box)
                dict[name]['labels'].append(1)
                dict[name]['scores'].append(float(tokens[-1]))

            #print(dict)
            with open(dst_path + filename.split('.')[0] + '.pkl', 'wb') as f: 
                pickle.dump(dict, f)

 

        else:
            continue

