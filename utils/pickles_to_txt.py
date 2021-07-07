import pickle
import os


if __name__ == '__main__':
    src_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/results/openpcdet/0_demo/pickles/'
    dst_path = 'D:/Ian/UNI/5_Master_CV/M9_TFM/results/openpcdet/0_demo/txt/'
    
    for filename in os.listdir(src_path):
        if filename.endswith(".pkl"):
            objects = []
            with (open(os.path.join(src_path,filename), "rb")) as openfile:
                while True:
                    try:
                        objects.append(pickle.load(openfile))
                    except EOFError:
                        break
            #print(objects[0])
            for key, value in objects[0].items():
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 h w l x y z heading score 
                # Pedestrian -1 -1 -10 -1 -1 -1 -1 dz dx dy y1 z1 x1 heading score 
                # [x, y, z, dx, dy, dz, heading]

                for i in range(0,len(value["boxes"])):
                    box = value["boxes"][i]
                    f = open(dst_path+"{}.txt".format(filename.split('.')[0]), "a+")
                    f.write("pedestrian -1 -1 -10 -1 -1 -1 -1 {} {} {} {} {} {} {} {}\n".format(box[5],box[3],box[4],box[1],box[2],box[0],box[6], value["scores"][i]))
                    f.close()
                    

        else:
            continue

