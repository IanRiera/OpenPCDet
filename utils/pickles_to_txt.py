import pickle

if __name__ == '__main__':
    results_path= 'D:\\Ian\\UNI\\5_Master_CV\\M9_TFM\\media\\beamagine\\dataset\\results\\pickles\\20200110_135503_976.pkl'
    objects = []
    with (open(results_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    for key, value in objects[0].items():
        
        for i in range(0,len(value)):
            print(objects[0])