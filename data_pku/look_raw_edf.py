from mne import annotations
import pandas as pd 
import numpy as np 
import os, sys  
import mne 
import re 

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
    #print(result)
def read_single_edf(path):
    print(path)
    #path = os.path.join(root_path,path)
    data = mne.io.read_raw_edf(path)
    raw_data,data_times = data.get_data(return_times=True)
    raw_df = data.to_data_frame()
    #raw_df = raw_df[need_channels]
    #print(raw)
    #data = raw.get_data()
    labels = data.annotations.onset
    duration = data.annotations.duration
    descriptions = data.annotations.description
    #print(type(raw_data),type(data_times),type(labels),type(duration),type(descriptions))
    print(raw_data.shape,data_times.shape,labels.shape,duration.shape,descriptions.shape)
    #print(raw_df.iloc[:,:10].head())
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
    decode_time_des = []
    for des in descriptions:
        print(des)
        try:
            des = anotation_decode(des)
        except:
            print(f'decode wrong, des is {des} ')
            continue
        #if re.match(r'.*[0-9]+.*',des):
        if re.match(r'^\d.*[0-9]+.*',des[1:]) or len(des)==1: # 第二个字符必须是数字
            decode_time_des.append(des)
        
    #print(f'len is {len(decode_des)} ')
    for i,des in enumerate(decode_time_des):
        #print(f'i is {i}, des is {des}, type is {type(des)} ')
        
        if des == '1':
            try:
                start_second_list.append(round(float(decode_time_des[i-1][1:])))
            except:
                print(f'failed to get label 1, des is {des} ')
        if des == '2':
            try:
                end_second_list.append(round(float(decode_time_des[i-1][1:])))
            except:
                print(f'failed to get label 2, des is {des} ')

    #print(start_second_list)
    #print(end_second_list)
    #print(type(start_second_list[0]))
    assert len(start_second_list) == len(end_second_list), " read seiz time wrong, start_second_list not equal to end_second_list, check it first ------- "
    for i in range(len(start_second_list)):
        seiz_time_df.loc[i] = [i,start_second_list[i],end_second_list[i]]
    print(seiz_time_df)
    new_df = reduce_frequency(new_df,frequency)
    return new_df, seiz_time_df
#print(path_list)

if __name__ == '__main__':
    #path = '/home/nanke/50_seizure_pku/1.E-210428张书航.edf'
    path = '/home/nanke/50_seizure_pku/E-210428张书航-num.edf'
    root_path = "/mnt/data1/share_data/pku_no1_0916/lvyahan"
    
    path = '/mnt/data1/share_data/pku_no1_0916/lvyahan/E-210465吕雅涵第一组发作间期2.edf'
    path = '/mnt/data1/share_data/pku_no1_0916/lvyahan/E-210465吕雅涵第一组发作期.edf'
    path = '/mnt/data1/share_data/pku_no1_0916/lvyahan/E-210465吕雅涵第一组发作间期2.edf'

    path = '/mnt/data1/share_data/70个发作期-北大/E-210294代晨旭.edf'
    path = '/mnt/data1/share_data/70个发作期-北大/E-210414张嘉棋2.edf'
    path = '/mnt/data1/share_data/70个发作期-北大/E-210329张梦瑶.edf'
    #path = '/home/nanke/50_seizure_pku/E-210432王晟楠-num.edf'
    read_single_edf(path)
    #read_edf(path_list[0])





