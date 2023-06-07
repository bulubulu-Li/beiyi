import pickle5 as pickle

import os
this_root_path = './beiyi/0601_single/pos_5'
path_list = os.listdir(this_root_path)
arr_list = []
for name in path_list:
    name_path = os.path.join(this_root_path,name)
    for path in os.listdir(name_path):
        abs_path = os.path.join(name_path, path)
        with open(abs_path, 'rb') as f:
            arr = pickle.load(f)
            arr.to_pickle(name_path)
            arr_list.append(arr)
    break
# all_arr = np.concatenate(arr_list, axis=0)
