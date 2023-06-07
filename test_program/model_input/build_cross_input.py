
import mne
import mne.annotations
import os,sys 
import numpy as np 
import pandas as pd 
import re, gc
import time 
import pickle
import tqdm 
import logging
sys.path.append('/home/nanke/pku1')
import torch 
from utils import set_cnn_model_parameters,use_cuda
from data_pku.pos_sample_scale import build_scale_together
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
# global set params
log_root_path = '/mnt/data1/user/kenan/log'

config = set_cnn_model_parameters()

seed = config.seed 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

np.random.seed(seed)

def reduce_df(df):
    start_mem = round(df.memory_usage().sum() / 1024 ** 3,3)
    for col in tqdm(df.columns):
        df[col] = df[col].astype(np.int8)
    end_mem = round(df.memory_usage().sum() / 1024 ** 3,3)
    #print(f'start mem is {start_mem} GB, end mem is {end_mem} GB')    

def look_up_df(path):
    df = pd.read_pickle(path)
    print(f'df shape is {df.shape} ')
    print(df.head())
    return df 

def get_train_data(name_list,flag):
    
    this_root_path = os.path.join(f'/mnt/data1/user/kenan/bei_yi_70/1112_single/{flag}_2')
    path_list = os.listdir(this_root_path)
    arr_list = []
    for name in path_list:
        if name not in name_list:
            continue
        name_path = os.path.join(this_root_path,name)
        for path in os.listdir(name_path):
            abs_path = os.path.join(name_path, path)
            with open(abs_path, 'rb') as f:
                arr = pickle.load(f)
                arr_list.append(arr)
    all_arr = np.concatenate(arr_list, axis=0)
    # print(all_arr.shape)
    return all_arr


def get_pretrain_data_old(train_data, test_data, pretrain_size, flag):
    print(f'in get {flag} pretrain data, train_data shape is {train_data.shape}, test data shape is {test_data.shape}, pretrain_size is {pretrain_size} ')
    np.random.shuffle(test_data)
    test_split_index = int(test_data.shape[0] * pretrain_size)
    new_train_data = np.concatenate((train_data, test_data[:test_split_index]), axis=0)
    new_test_data = test_data[test_split_index:]
    print(f'after use {flag} pretrain data, new_train_data shape is {new_train_data.shape}, new_test_data shape is {new_test_data.shape} ')
    return new_train_data, new_test_data

def get_pretrain_data(test_data, pretrain_size, flag):
    print(f'in get {flag} pretrain data, test data shape is {test_data.shape}, pretrain_size is {pretrain_size} ')
    np.random.shuffle(test_data)
    test_split_index = int(test_data.shape[0] * pretrain_size)
    new_train_data = test_data[:test_split_index]
    new_test_data = test_data[test_split_index:]
    print(f'after use {flag} pretrain data, new_train_data shape is {new_train_data.shape}, new_test_data shape is {new_test_data.shape} ')
    return new_train_data, new_test_data


class CNN_INPUT_cross():
    def __init__(self,config):
        self.config = config 
        self.train_test_ratio = 0.8
        self.train_eval_ratio = 0.8
        self.pos_train_scale = config.use_scale
        self.use_pretrain = config.use_pretrain

        self.pos_data_path = '/mnt/data1/user/kenan/bei_yi_70/1107/pos_2_all.pkl'
        self.neg_data_path = '/mnt/data1/user/kenan/bei_yi_70/1107/neg_2_all.pkl'

        self.train_name_list = ['陈康霖', '郑桓宇', '白靖宇', '王震林', '谢景恬', 
                        '张念晨', '冯雨欣', '王一回', '吕雅涵', '卢一赫', 
                        '孙嘉豪', '陈宇轩', '卢婧之', '宋钰苇', '柳叶恩', 
                        ]
        self.test_name_list = ['耿子卓', '孙奥发', '黄嘉琪', '陈柳妍']

        self.load_raw_data()

    def load_raw_data(self):
        
        train_name_list = ['陈康霖', '郑桓宇', '白靖宇', '王震林', '谢景恬', 
                        '张念晨', '冯雨欣', '王一回', '吕雅涵', '卢一赫', 
                        '孙嘉豪', '陈宇轩', '卢婧之', '宋钰苇', '柳叶恩', 
                        ]
        test_name_list = ['耿子卓', '孙奥发', '黄嘉琪', '陈柳妍']

        all_name_list = train_name_list + test_name_list
        name_num = len(all_name_list)

        all_index = list(range(name_num))
        test_index = list(np.random.choice(name_num, size=4))
        for index in test_index:
            pass 
            #all_index.remove(index)
        train_index = all_index

        #train_name_list = all_name_list[train_index]
        #test_name_list = all_name_list[test_index]
        print(f'train_name_list is {train_name_list} ')
        print(f'test_name_list is {test_name_list} ')

        pos_train_data = get_train_data(train_name_list, 'pos')
        neg_train_data = get_train_data(train_name_list, 'neg')
        # print(f'data 1', pos_train_data.shape.shape, neg_train_data.shape )
        
        pos_test_data = get_train_data(test_name_list, 'pos')
        neg_test_data = get_train_data(test_name_list, 'neg')
        # print(f'data 2', pos_test_data.shape, neg_test_data.shape)
        # get pretrain test data
        if self.use_pretrain>0:
            pos_train_data, pos_test_data = get_pretrain_data(pos_train_data, pos_test_data, pretrain_size=self.use_pretrain, flag='pos')
            neg_train_data, neg_test_data = get_pretrain_data(neg_train_data, neg_test_data, pretrain_size=self.use_pretrain, flag='neg')

        # split train/eval data
        pos_train_data, pos_eval_data = train_test_split(pos_train_data, test_size=0.2, random_state=42)
        neg_train_data, neg_eval_data = train_test_split(neg_train_data, test_size=0.2, random_state=42)

        all_scale_way_list = ['add','sub','add_sub','mutiply','random'] * 5
        scale_way_list = all_scale_way_list[:self.pos_train_scale]
        if self.pos_train_scale > 0:
            print(f'use scale is {self.pos_train_scale}, scale_way_list is {scale_way_list} ')
            pos_train_data = build_scale_together(pos_train_data, scale_way_list)


        self.train_loader = self.build_data_iterator(pos_train_data, neg_train_data, 'train')
        self.eval_loader = self.build_data_iterator(pos_eval_data, neg_eval_data, 'eval')
        self.test_loader = self.build_data_iterator(pos_test_data, neg_test_data, 'test')


    def build_data_iterator(self, pos_arr, neg_arr, flag):
        print(f'------ {flag} iterator begin ')
        pos_y = np.ones(pos_arr.shape[0])
        neg_y = np.zeros(neg_arr.shape[0])
        
        x_arr = np.concatenate((pos_arr, neg_arr), axis=0)
        y_arr = np.concatenate((pos_y, neg_y), axis=0)
        print(f'build {flag} iterator here, pos num is {pos_arr.shape[0]}, neg_num is {neg_arr.shape[0]} ')

        tensor_dataset = TensorDataset(
            torch.from_numpy(x_arr),
            torch.from_numpy(y_arr)
        )
        shuffle = True
        if flag == 'test':
            shuffle = False
        data_loader = DataLoader(dataset=tensor_dataset,
            shuffle=shuffle,
            batch_size = self.config.batch_size,
            drop_last = True
        )
        return data_loader


class CNN_INPUT_cross_fine_tune():
    def __init__(self, config):
        self.config = config 

        self.train_test_ratio = 0.8
        self.train_eval_ratio = 0.8
        self.pos_train_scale = config.use_scale
        self.use_pretrain = 0.2 #config.use_pretrain

        self.train_name_list = ['陈康霖', '郑桓宇', '白靖宇', '王震林', '谢景恬', 
                        '张念晨', '冯雨欣', '王一回', '吕雅涵', '卢一赫', 
                        #'孙嘉豪', '陈宇轩', '卢婧之', '宋钰苇', '柳叶恩', 
                        ]
        self.test_name_list = ['耿子卓', '孙奥发', '黄嘉琪', '陈柳妍','孙嘉豪', '陈宇轩', '卢婧之', '宋钰苇', '柳叶恩', ]
        self.test_num = len(self.test_name_list)

        self.load_train_data()

    def load_train_data(self):
        train_name_list = self.train_name_list
        print(f'train_name_list is {train_name_list} ')
        
        pos_train_data = get_train_data(train_name_list, 'pos')
        neg_train_data = get_train_data(train_name_list, 'neg')
        
        # split train/eval data
        pos_train_data, pos_eval_data = train_test_split(pos_train_data, test_size=0.2, random_state=42)
        neg_train_data, neg_eval_data = train_test_split(neg_train_data, test_size=0.2, random_state=42)

        all_scale_way_list = ['add','sub','add_sub','mutiply','random'] * 5
        scale_way_list = all_scale_way_list[:self.pos_train_scale]
        if self.pos_train_scale > 0:
            print(f'use scale is {self.pos_train_scale}, scale_way_list is {scale_way_list} ')
            pos_train_data = build_scale_together(pos_train_data, scale_way_list)

        self.train_loader = self.build_data_iterator(pos_train_data, neg_train_data, 'train')
        self.eval_loader = self.build_data_iterator(pos_eval_data, neg_eval_data, 'eval')
        

    def load_test_data(self, patient_id=0):
        test_name_list = self.test_name_list[patient_id:patient_id+1]
        print(f'test_name_list is {test_name_list} ')
        pos_test_data = get_train_data(test_name_list, 'pos')
        neg_test_data = get_train_data(test_name_list, 'neg')

        all_scale_way_list = ['add','sub','add_sub','mutiply','random'] * 5
        scale_way_list = all_scale_way_list[:self.pos_train_scale]

        pos_fine_tune_data, pos_test_data = get_pretrain_data(pos_test_data, pretrain_size=self.use_pretrain, flag='pos')
        if self.pos_train_scale > 0:
            print(f'use scale is {self.pos_train_scale}, scale_way_list is {scale_way_list} ')
            pos_fine_tune_data = build_scale_together(pos_fine_tune_data, scale_way_list)

        neg_fine_tune_data, neg_test_data = get_pretrain_data(neg_test_data, pretrain_size=self.use_pretrain, flag='neg')

        test_loader = self.build_data_iterator(pos_test_data, neg_test_data, 'test')
        fine_tune_loader = self.build_data_iterator(pos_fine_tune_data, neg_fine_tune_data, 'fine_tune')
        return fine_tune_loader, test_loader

    def build_data_iterator(self, pos_arr, neg_arr, flag):
        print(f'------ {flag} iterator begin ')
        pos_y = np.ones(pos_arr.shape[0])
        neg_y = np.zeros(neg_arr.shape[0])
        
        x_arr = np.concatenate((pos_arr, neg_arr), axis=0)
        y_arr = np.concatenate((pos_y, neg_y), axis=0)
        print(f'build {flag} iterator here, pos num is {pos_arr.shape[0]}, neg_num is {neg_arr.shape[0]} ')

        tensor_dataset = TensorDataset(
            torch.from_numpy(x_arr),
            torch.from_numpy(y_arr)
        )
        shuffle = True
        if flag == 'test':
            shuffle = False
        data_loader = DataLoader(dataset=tensor_dataset,
            shuffle=shuffle,
            batch_size = self.config.batch_size,
            drop_last = True
        )
        return data_loader

if __name__ == '__main__':

    #CNN_Data = CNN_INPUT_cross(config)
    CNN_Data = CNN_INPUT_cross_fine_tune(config)
    CNN_Data.load_test_data()






