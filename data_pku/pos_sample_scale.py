
import pandas as pd 
import numpy as np 
import gc,os
import random
import pickle 
import sys

sys.path.append('/home/nanke/pku1')
from utils import set_cnn_model_parameters

global_cnn_config = set_cnn_model_parameters()
time_window_size = global_cnn_config.time_window_size 
frequency = global_cnn_config.frequency 
dataset = global_cnn_config.dataset 

print(f'time_window_size is {time_window_size}, dataset is {dataset} ')

need_19_channels = ['EEG Fp1-Ref', 'EEG Fp2-Ref', 'EEG F3-Ref', 'EEG F4-Ref', 'EEG C3-Ref', 
    'EEG C4-Ref', 'EEG P3-Ref', 'EEG P4-Ref', 'EEG O1-Ref', 'EEG O2-Ref', 
    'EEG F7-Ref', 'EEG F8-Ref', 'EEG T3-Ref', 'EEG T4-Ref', 'EEG T5-Ref', 
    'EEG T6-Ref', 'EEG Fz-Ref', 'EEG Cz-Ref', 'EEG Pz-Ref'
 ]



def look_up_df(df):
    print(f'look up, df shape is {df.shape} ')
    #print(df.iloc[:,-5::].describe())
    #print(df.dtypes)
    print(df.iloc[:,-5::].head())

def scale_df_once(df,mode='add'):
    new_df = pd.DataFrame()
    for col in df.columns:
        if mode=='add':
            new_df[col] = df[col].apply(lambda x:x+1 ).astype(np.int16)
        elif mode=='sub':
            new_df[col] = df[col].apply(lambda x:x-1 ).astype(np.int16)
        elif mode=='add_sub':
            new_df[col] = df[col].apply(lambda x:x+1 if random.randint(0,100)%2==0 else x-1).astype(np.int16)
        elif mode=='mutiply':
            new_df[col] = df[col].apply(lambda x:x*1.1 if x%2==0 else x*0.9).astype(np.int16)
        elif mode=='random':
            new_df[col] = df[col].apply(lambda x:x + int(random.uniform(-3,3)) ).astype(np.int16)
    #print(f'test new df ')
    #look_up_df(new_df)
    return new_df

def build_scale_together(pos_sample_array,scale_way_list,dataset='mit'):
    if dataset == 'pku':
        frequency = 50
    elif dataset == 'mit':
        frequency = 64 

    print(f'build scale together, dataset is {dataset}, scale way list is {scale_way_list} ')
    all_sample_array_list = []
    for k in range(len(scale_way_list)):
        print(f'now scale way is {scale_way_list[k]}, scale begin ')
        for i in range(pos_sample_array.shape[0]):
            if k==0:
                all_sample_array_list.append(pos_sample_array[i]) # add itself
                #print(f'pos_sample_array[i] shape is {pos_sample_array[i].shape} ')
            #tmp_df = pd.DataFrame(pos_sample_array[i],columns=need_19_channels)
            tmp_df = pd.DataFrame(pos_sample_array[i])
            new_df = scale_df_once(tmp_df,mode=scale_way_list[k])
            if new_df.shape[0]!=frequency*time_window_size:
                print(f'new df shape wrong ,shape is {new_df.shape} ---------- ')
                continue
            all_sample_array_list.append(new_df.values)
        print(f'all_sample_array_list len is {len(all_sample_array_list)} ')
    final_scaled_pos_array = np.array(all_sample_array_list)
    print(f'final_scaled_pos_array shape is {final_scaled_pos_array.shape} ')
    return final_scaled_pos_array


def run_scale():
    pos_sample_path = f'/mnt/data1/user/kenan/bei_yi/0918/吕雅涵/pos_2/all_2arr.pkl'
    with open(pos_sample_path,'rb') as f:
        pos_sample_array = pickle.load(f)
        print(f'before scale, pos_sample_array shape is {pos_sample_array.shape}  ')
#new_array = build_scale(pos_sample_array)

#scale_way_list = ['add','sub','add_sub','mutiply','random']
    
    pos_scale_array = build_scale_together(pos_sample_array,scale_way_list)
    print(f'pos_scale_array shape is {pos_scale_array.shape} ')
    #continue
    return 
    pos_scale_save_path = pos_sample_path = ''
    with open(pos_scale_save_path,'wb') as f:
        pickle.dump(pos_scale_array,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(f'scale array to pkl, save path is {pos_scale_save_path} ')


if __name__ == '__main__':
    #patient_id_list = test_id
    scale_way_list = ['add','sub','mutiply','random','add_sub']
    scale_rate = len(scale_way_list)
    run_scale()




