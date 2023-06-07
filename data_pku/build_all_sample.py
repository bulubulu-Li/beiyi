from posixpath import abspath
from mne import annotations
import pandas as pd 
import numpy as np 
import os, sys  
import mne 
import re 
import math
import gc 
import pickle
import random 
import json 

def read_json(path):
    with open(path,'r') as f:
        js = json.load(f)
        #print(js)
        return js

sys.path.append('/home/nanke/pku1')
from utils import set_cnn_model_parameters
params = set_cnn_model_parameters()

pd.set_option('display.max_columns', 50)

need_19_channels = ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref', 'EEG C3-Ref', 
    'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref', 'EEG O1-Ref', 'EEG O2-Ref', 
    'EEG F7-Ref', 'EEG F8-Ref', 'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 
    'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref'
 ]

need_21_channels = ['time', 'time_second', 'EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref', 'EEG C3-Ref', 
    'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref', 'EEG O1-Ref', 'EEG O2-Ref', 
    'EEG F7-Ref', 'EEG F8-Ref', 'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 
    'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref'
 ]


def anotation_decode(x,en = 'iso-8859-1',de = 'gbk'):
    return (x.encode(en)).decode(de)

def add_timestamp(df,frequency):
    N = df.shape[0]
    total_seconds = int(N/frequency)
    df['time_second'] = df['time'].apply(lambda x:int(x*0.5/frequency) )
    #print('show new df time ')
    #print(df[['time','time_second']].iloc[495:505,:].head(10))
    #print(df[['time','time_second']].iloc[:,:].tail(10))
    return df 

def reduce_frequency(raw_df,frequency):
    reduce_rate = 10
    start_mem = round(raw_df.memory_usage().sum() / 1024 ** 3,3)
    #print(raw_df.iloc[:,:5].head())
    #print(raw_df.iloc[:,:5].tail(12))
    #print(f'---- ---- ----')
    print(f'origin frequency is {frequency}, final frequency is {frequency/reduce_rate} ')
    new_df = raw_df[raw_df.index%reduce_rate==0]
    #print(new_df.iloc[:,:5].tail(12))
    end_mem = round(new_df.memory_usage().sum() / 1024 ** 3,3)
    print(f'raw df shape is {raw_df.shape}, new df shape is {new_df.shape} ')
    print(f'start memory is {start_mem} GB, after squeeze memory is {end_mem} GB')
    return new_df

def read_single_edf(edf_path):
    print(edf_path)
    #path = os.path.join(root_path,path)
    data = mne.io.read_raw_edf(edf_path)
    raw_data,data_times = data.get_data(return_times=True)
    raw_df = data.to_data_frame()
    #print(raw)
    #data = raw.get_data()
    labels = data.annotations.onset
    duration = data.annotations.duration
    descriptions = data.annotations.description
    print(type(raw_data),type(data_times),type(labels),type(duration),type(descriptions))
    print(raw_data.shape,data_times.shape,labels.shape,duration.shape,descriptions.shape)
    print(raw_df.iloc[:,:10].head())
    #return raw_data, data_times 
    print(f'df shape is {raw_df.shape} ')
    #print(raw_df.iloc[:,:10].tail())

    info = data.info
    channels = data.ch_names
    frequency = info['sfreq']
    print(f'frequency is {frequency} ')
    #print(f'info are {info} ')
    #print(f'channels are {channels} ')
    #print(f'labels are {labels}  ')
    #print(f'descriptions are {descriptions} \n ')
    new_df = add_timestamp(raw_df,frequency)
    new_df = new_df[need_21_channels]

    seiz_time_df = pd.DataFrame(columns=['seiz_id','seiz_start_second','seiz_end_second'])
    start_second_list = []
    end_second_list = []
    decode_des = []
    for des in descriptions:
        des = anotation_decode(des)
        if re.match(r'.*[0-9]+.*',des):
            decode_des.append(des)
        #print(des)
    #print(f'len is {len(decode_des)} ')
    for i,des in enumerate(decode_des):
        #print(f'i is {i}, des is {des}, type is {type(des)} ')
        if des == '1':
            start_second_list.append(round(float(decode_des[i-1][1:])))
        if des == '2':
            end_second_list.append(round(float(decode_des[i-1][1:])))
    print(start_second_list)
    print(end_second_list)
    #print(type(start_second_list[0]))
    assert len(start_second_list) == len(end_second_list), " read seiz time wrong, check it first ------- "
    for i in range(len(start_second_list)):
        seiz_time_df.loc[i] = [i,start_second_list[i],end_second_list[i]]
    print(seiz_time_df)
    new_df = reduce_frequency(new_df,frequency)
    return new_df, seiz_time_df



def build_single_pos_sample(raw_df, seiz_time_df, date, patient_name):
    # build_pos_sample from single edf file
    time_window_size = params.time_window_size # 
    pos_sample_list = []

    root_path = f'/mnt/data1/user/kenan/bei_yi/{date}/{patient_name}/pos_{time_window_size}'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    path_list = os.listdir(root_path)
    for path in path_list:
        if 'all' in path_list:
            print(f'all pos array already done, all path is {path}, return ')
            return 
    num = len(path_list)
    pos_save_path = os.path.join(root_path,f'num{num}.pkl')
    if os.path.exists(pos_save_path):
        return 
    
    for i in range(seiz_time_df.shape[0]):
        seiz_start_second = seiz_time_df.at[i,'seiz_start_second']
        seiz_end_second = seiz_time_df.at[i,'seiz_end_second']
        print(f'test pos,seiz_start_second is {seiz_start_second}, seiz_end_second is {seiz_end_second}')
            # slide window begin
        window_start_time = seiz_start_second
        window_end_time = window_start_time + time_window_size 
        cnt = 1
        while window_end_time < seiz_end_second:
            
            # key step , attention 
            tmp_df = raw_df.loc[(raw_df['time_second']>=window_start_time) & (raw_df['time_second']<window_end_time)]
            if cnt%20==0:
                #print(f'seiz_start_second is {seiz_start_second}, seiz_end_second is {seiz_end_second}')
                print(f'window_start_time is {window_start_time}, window_end_time is {window_end_time} ')
                print(f' **** check 1, tmp_df shape is {tmp_df.shape}, cnt is {cnt} ')

            if len(pos_sample_list)<0: #and (pos_sample_list[-1].shape[0]!=640 and pos_sample_list[-1].shape[1]!=22):
                    print(f'pos sample wrong wrong wrong ,attention and attention ------------------------------')
                    pos_sample_list.pop()
                #update time
            window_start_time = window_start_time + 1  #time_window_size # slide to next 
            window_end_time = window_start_time + time_window_size       # keep time interval
            #print(f'test 1,i is {i}, cnt is {cnt}, window sliding  ')
            cnt += 1
            #print(tmp_df.iloc[:,-5::].head())
            #pos_sample_list.append( tmp_df.drop(['time','time_second'],axis=1).values.astype(np.int16))
            pos_sample_list.append(tmp_df[need_19_channels].values.astype(np.int16))

            del tmp_df
            gc.collect()
    
    pos_train_data_array = np.array(pos_sample_list).astype(np.int16)
    print(f'pos shape is {pos_train_data_array.shape} ')
    with open(pos_save_path,'wb') as f:
        pickle.dump(pos_train_data_array,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'save pos sample to path {pos_save_path} done ')

    with open(pos_save_path,'rb') as f:
        pos_array = pickle.load(f)
        print(f'final check, pos array shape is {pos_array.shape} ')

def check_neg_time_valid(seiz_time_df,tmp_start_index,tmp_end_index):
    # 四种情况： 1. 一端重叠    2. 全在里面
    valid_flag = 1
    for i in range(seiz_time_df.shape[0]):
        seiz_start_second = seiz_time_df.at[i,'seiz_start_second'] - 5
        seiz_end_second = seiz_time_df.at[i,'seiz_end_second'] + 5
        if tmp_end_index>=seiz_start_second and tmp_end_index <= seiz_end_second:
            valid_flag = 0 # 
        if tmp_start_index >= seiz_start_second and tmp_start_index <= seiz_end_second:
            valid_flag = 0
        if tmp_start_index>=seiz_start_second and tmp_end_index <=seiz_end_second:
            valid_flag = 0
        if tmp_start_index<=seiz_start_second and tmp_end_index >=seiz_end_second:
            valid_flag = 0 
    if valid_flag==0:
         return False
    return True

def build_single_neg_sample(raw_df, seiz_time_df, date, patient_name):
    # build_pos_sample from single edf file
    time_window_size = params.time_window_size # 2
    neg_sample_list = []

    root_path = f'/mnt/data1/user/kenan/bei_yi/{date}/{patient_name}/neg_{time_window_size}'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    path_list = os.listdir(root_path)
    for path in path_list:
        if 'all' in path_list:
            print(f'all neg array already done, all path is {path}, return ')
            return 
    num = len(path_list)
    neg_save_path = os.path.join(root_path,f'num{num}.pkl')
    
    neg_pos_rate = 4
    pos_sample_num = 50
    need_neg_sample_num = pos_sample_num * neg_pos_rate

    neg_sample_cnt = 0
    cnt = 0
    edf_start_second = 0
    edf_end_second = raw_df.shape[0] / 50 # 50 is the frequency 
    print(f'edf_end_second is {edf_end_second} ')
    while neg_sample_cnt < need_neg_sample_num:
        cnt += 1
        tmp_start_index = random.randint(edf_start_second,edf_end_second)
        tmp_end_index = tmp_start_index + time_window_size
        if cnt%20==0:
            print(f'cnt is {cnt} , neg sample cnt is {neg_sample_cnt}, tmp_start_index is {tmp_start_index}, tmp_end_index is {tmp_end_index} ')
        
        if check_neg_time_valid(seiz_time_df,tmp_start_index,tmp_end_index) == False:
                continue        # when index in seiz df, return false . discard
        else:
            tmp_df = raw_df.loc[(raw_df['time_second']>=tmp_start_index) & (raw_df['time_second']<tmp_end_index)]
            if tmp_df.shape[0]==0:
                continue
            if len(neg_sample_list)>1 and (tmp_df.shape[0]!=neg_sample_list[-1].shape[0]):
                print(f'tmp shape wrong, shape is {tmp_df.shape}, neg_sample_list[-1] shape is {neg_sample_list[-1].shape} ')
                continue
            neg_sample_list.append(tmp_df[need_19_channels].values.astype(np.int16))
            # update 
            neg_sample_cnt += 1
            del tmp_df
            gc.collect()
        
        #break

    #return 
    neg_train_data_array = np.array(neg_sample_list).astype(np.int16)
    print(f'neg shape is {neg_train_data_array.shape} ')
    with open(neg_save_path,'wb') as f:
        pickle.dump(neg_train_data_array,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'save neg sample to path {neg_save_path} done ')
        
    with open(neg_save_path,'rb') as f:
        neg_array = pickle.load(f)
        print(f'final check, neg array shape is {neg_array.shape} ')

def get_edf_paths(root_path):
    path_list = os.listdir(root_path)
    pos_paths = []
    neg_paths = []
    for path in path_list:
        #print(path)
        abs_path = os.path.join(root_path,path)
        if r'间' in path:
            neg_paths.append(abs_path)
        else:
            pos_paths.append(abs_path)
    return pos_paths, neg_paths


def deal_single_edf(edf_path, date, patient_name, flag):
    raw_df, seiz_time_df = read_single_edf(edf_path)
    if flag == 'pos':
        build_single_pos_sample(raw_df, seiz_time_df, date, patient_name)
    else:
        build_single_neg_sample(raw_df, seiz_time_df, date, patient_name)

def merge_all_edfs(date, patient_name, flag):
    time_window_size = params.time_window_size
    root_path = f'/mnt/data1/user/kenan/bei_yi/{date}/{patient_name}/{flag}_{time_window_size}'
    path_list = os.listdir(root_path)
    for path in path_list:
        if 'all' in path_list:
            print(f'all {flag} array already done, save path is {path}, return ')
            return 

    nums = len(path_list)
    arr_list = []
    for i,path in enumerate(path_list):
        abs_path = os.path.join(root_path,path)
        with open(abs_path,'rb') as f:
            tmp_arr = pickle.load(f)
            print(f'i is {i}, tmp arr shape is {tmp_arr.shape} ')
        arr_list.append(tmp_arr)
    final_arr = np.concatenate(arr_list)
    print(f'final arr shape is {final_arr.shape} ')
    merge_path = os.path.join(root_path,f'all_{nums}arr.pkl')
    with open(merge_path,'wb') as f:
        pickle.dump(final_arr,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'merge path save to {merge_path} ')


def deal_single_patient(patient_name, patient_root_path):
    pos_paths, neg_paths = get_edf_paths(patient_root_path)
    #neg_path = '/mnt/data1/share_data/pku_no1_0916/lvyahan/E-210465吕雅涵第一组发作间期2.edf'
    #pos_path = '/mnt/data1/share_data/pku_no1_0916/lvyahan/E-210465吕雅涵第一组发作期.edf'
    date = '0918'

    for pos_path in pos_paths:
        pass 
        #deal_single_edf(pos_path, date, patient_name, 'pos')
    #merge_all_edfs(date, patient_name, 'pos')

    for neg_path in neg_paths:
        deal_single_edf(neg_path, date, patient_name, 'neg')
    #merge_all_edfs(date, patient_name, 'neg')

def merge_test():
    date = '0918'
    patient_name = '吕雅涵'

    merge_all_edfs(date, patient_name, 'pos')
    merge_all_edfs(date, patient_name, 'neg')

def deal_all_patients():
    patient_name = '吕雅涵'
    patient_root_path = '/mnt/data1/share_data/pku_no1_0916/lvyahan'
    deal_single_patient(patient_name, patient_root_path)
    merge_test()

if __name__ == '__main__':

    deal_all_patients()








