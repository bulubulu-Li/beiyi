from mne import annotations
import pandas as pd 
import numpy as np 
import os 
import mne 

def decoder(x,en = 'iso-8859-1',de = 'gbk'):
    return (x.encode(en)).decode(de)


def read_edf(path):
    # path = os.path.join(root_path,path)
    data = mne.io.read_raw_edf(path)
    raw_data,data_times = data.get_data(return_times=True)
    print(f'raw_data shape is {raw_data.shape} ')
    #print(raw)
    #data = raw.get_data()
    labels = data.annotations.onset
    duration = data.annotations.duration
    descriptions = data.annotations.description
    print(type(raw_data),type(data_times),type(labels),type(duration),type(descriptions))
    print(raw_data.shape,data_times.shape,labels.shape,duration.shape,descriptions.shape)
    minutes = raw_data.shape[1] /(500*60)
    print(f'minutes is {minutes} ')

    info = data.info
    channels = data.ch_names
    print(f'channels are {channels} ')
    #print(f'labels are {labels}  ')
    #print(f'descriptions are {descriptions} \n ')
    print(f'info are {info} ')
    result = []
    for des in descriptions:
        print(decoder(des))
        result.append(decoder(des))
    #print(result)

#print(path_list)

if __name__ == '__main__':
    #path = '/home/nanke/50_seizure_pku/1.E-210428张书航.edf'
    # path = '/home/nanke/50_seizure_pku/E-210428张书航-num.edf'
    # root_path = "/mnt/data1/share_data/pku_no1_0916/lvyahan"
    
    # path = '/mnt/data1/share_data/pku_no1_0916/lvyahan/E-210465吕雅涵第一组发作间期2.edf'
    # path = '/mnt/data1/share_data/pku_no1_0916/lvyahan/E-210465吕雅涵第一组发作期.edf'
    path = 'E-210569loujiawei.edf'
    #path = '/mnt/data1/share_data/pku_no1_0916/lvyahan/E-210465吕雅涵第一组发作间期2.edf'
    #path = '/home/nanke/50_seizure_pku/E-210432王晟楠-num.edf'
    read_edf(path)
    #read_edf(path_list[0])





