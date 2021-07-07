import pickle

if __name__ == '__main__':
    all_pickle_path = 'D:/Ian/UNI/5_Master_CV/M9_TFM/results/openpcdet/0_demo/000008_pred_pv_rcnn.pkl'
    example_path = 'D:/Ian/UNI/5_Master_CV/M9_TFM/results/openpcdet/0_demo/000008_pred_pv_rcnn.pkl'
    results_path= 'D:/Ian/UNI/5_Master_CV/M9_TFM/results/openpcdet/0_demo/pickles/'
    objects = []
    example = []

    with (open(all_pickle_path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    with (open(example_path, "rb")) as openfile:
        while True:
            try:
                example.append(pickle.load(openfile))
            except EOFError:
                break
    print(example[0])

    for key, value in objects[0].items():
        dict = {}
        name = key.split('/')[-1]
        dict[name] = value
        with open(results_path + name.split('.')[0] + '.pkl', 'wb') as f:
            pickle.dump(dict, f)