import os,sys 
import time 
sys.path.append('/home/nanke/pku1')
from utils import set_cnn_model_parameters
params = set_cnn_model_parameters()

log_root_path = '/mnt/data1/user/kenan/log'

start_time_tag = params.time 
if len(start_time_tag)<5:
    start_time_tag = time.strftime("%m%d%H%M")

path_list = os.listdir(log_root_path)

#print(len(path_list))

log_path = ''
for path in path_list:
    if start_time_tag in path:
        log_path = os.path.join(log_root_path,path)
        print(log_path)
    
if len(log_path)==0:
    print(f'time tag is {start_time_tag}, log path not found')
