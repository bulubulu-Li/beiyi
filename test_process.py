import os,sys 
import numpy as np 
import pandas as pd 
import pickle
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import time
import copy 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from utils import specificity_score, get_confusion_matrix
# print('啦啦啦')
# sys.path.append('/home/zhangzongpeng/pku1')
from tools.logger import Logger
# from model_input.build_cross_input import CNN_INPUT_cross, CNN_INPUT_cross_identity, CNN_INPUT_cross_fine_tune,CNN_INPUT_cross_gaussian
from model_input.build_cross_input import CNN_INPUT_cross, CNN_INPUT_cross_fine_tune


#from model_input.build_cross_input_mit import CNN_INPUT_cross, CNN_INPUT_cross_identity, CNN_INPUT_cross_fine_tune,CNN_INPUT_cross_gaussian

from utils import set_cnn_model_parameters,use_cuda, evaluate_result
# from models.model_cw_srnet_DANN import cw_srnet
from model_cw_srnet_DANN import cw_srnet

params = set_cnn_model_parameters()
time_window_size = params.time_window_size

# CNN_Data = CNN_INPUT_cross_gaussian(params)#CNN_INPUT_cross_fine_tune(params)

CNN_Data = CNN_INPUT_cross_fine_tune(params)#CNN_INPUT_cross_fine_tune(params)


# CNN_Data_identity = CNN_INPUT_cross_identity(params)

# global set params

#start_time_tag = time.strftime("%m%d%H%M%S")
start_time_tag = params.time 
if len(start_time_tag)<5:
    start_time_tag = time.strftime("%m%d%H%M")
model_root_path = '/mnt/data1/user/zzp/result/model_save/beiyi/'
if not os.path.exists(model_root_path):
    os.makedirs(model_root_path)


log_root_path = './log'
dataset = params.dataset
model_name = 'pretrain_cnn'
log_path = os.path.join(log_root_path, f'{dataset}_{model_name}_tag{start_time_tag}.log')
log = Logger(log_path)
log.logger.info(f'log path is {log_path} ')

EPOCHS = params.epochs
BATCH_SIZE = params.batch_size  #32 #params.batch_size
FREQUENCY = params.frequency 

#use_cw_block = params.use_cw_block

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = params.model_id

model_name = params.model_name 


log.logger.info(f'ResNet18 model version is  v{model_id} ')
log.logger.info(f'params are \n {params} ')
# model params

loss_function = nn.BCELoss()
loss_funct_identity=nn.CrossEntropyLoss()

# get data 
#CNN_Data = CNN_INPUT_cross(params)

time_window_size = params.time_window_size # 

def evaluate(model,eval_iterator,params,epoch_i,run_mode='eval'):
    log.logger.info(f'-- epoch is {epoch_i}, run_mode is {run_mode}, begin -- ')
    model.eval()
    eval_loss = 0.0
    label_list = []
    predict_list = []
    res_list = []
    for i,(batch_x,batch_y) in enumerate(eval_iterator):
        batch_x = batch_x[:,:,:-1] # [batch_size,128,19-1]
        # batch_x = use_cuda(batch_x.float())
        # batch_y = use_cuda(batch_y.float())
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        print(batch_x.shape)
        model_input = batch_x.view(BATCH_SIZE,1,FREQUENCY*time_window_size,-1)
        # print(model_input.shape)
        predict_y = model(model_input)#无法输出结果
        # print(predict_y)
        # exit()
        loss = loss_function(predict_y,batch_y)
        eval_loss += loss.item()
        label_list.extend(batch_y.reshape(-1).tolist())
        predict_list.extend(predict_y.data.cpu().tolist())
        
    for i in range(len(predict_list)):
        res_list.append((predict_list[i],label_list[i]))
    # print(res_list)
    # cnt_7 = 0
    # cnt_7_pos = 0     
    # for i in range(len(res_list)):
    #     if res_list[i][0]>=0.5:
    #         cnt_7+=1
    #         if res_list[i][1]==1.0:
    #             cnt_7_pos+=1
    # print(cnt_7,cnt_7_pos,len(res_list))
    predict_label = [1 if p>params.threshold else 0 for p in predict_list]
    #check_result = [(label_list[i],predict_label[i]) for i in range(100)]
    accuracy = round(accuracy_score(label_list,predict_label),4)
    precision = round(precision_score(label_list,predict_label),4)
    recall = round(recall_score(label_list,predict_label),4)
    F1_score = round(f1_score(label_list,predict_label),4)
    eval_auc = round(roc_auc_score(label_list,predict_list),4)
    specificity = round(specificity_score(label_list,predict_label),4)
    log.logger.info(f'run mode is {run_mode}, epoch i is {epoch_i}, {run_mode} auc is {eval_auc} , {run_mode} loss is {eval_loss} ')
    log.logger.info(f'accuracy is {accuracy}, --- {run_mode} recall is {recall} ---,  --- {run_mode} precision is {precision} ---, f1 score is {F1_score}, show confusion matrix below ')
    #get_confusion_matrix(label_list, predict_label)
    # print(precision)

    return accuracy,precision,recall,specificity,eval_auc, eval_loss




from utils import get_name_pinyin
def run_test_once(best_model):
    optimizer = torch.optim.Adam(best_model.parameters(),lr=params.lr)
    test_num = CNN_Data.test_num 
    test_result_df = pd.DataFrame(columns=['patient','Sensitivity','Specificity','Accuracy','AUC']) 
    for i in range(test_num):
        fine_tune_loader, test_loader = CNN_Data.load_test_data(patient_id=i)#关键是获取数据
        patient_name = CNN_Data.test_name_list[i]
        # name_py = get_name_pinyin(patient_name)
        # print(f'i is {i}, name is {patient_name}, py is {name_py} ')
        #tmp_model = best_model #
        tmp_model = copy.deepcopy(best_model)
        tmp_model.best_epoch = i
        
        # test_result_dic = {}
        # accuracy,precision,recall,specificity,train_auc, train_loss, fine_tune_model = train_once(model, optimizer, fine_tune_loader, params,i)
        # fine_tune_epochs = 10
        # best_fine_model = copy.deepcopy(fine_tune_model)
        # best_train_score = recall * 1.5 + specificity + train_auc
        # for k in range(fine_tune_epochs):
        #     accuracy,precision,recall,specificity,train_auc, train_loss, fine_tune_model = train_once(fine_tune_model, optimizer, fine_tune_loader, params,i)
        #     tmp_score = recall * 1.5 + specificity + train_auc
        #     if tmp_score > best_train_score:
        #         best_train_score = tmp_score
        #         best_fine_model = copy.deepcopy(fine_tune_model)

        # test_accuracy,test_precision,test_sensitivity,test_specificity,test_auc, _ = evaluate(best_fine_model,test_loader,params,run_mode='test',epoch_i=i)
        test_accuracy,test_precision,test_sensitivity,test_specificity,test_auc, _ = evaluate(best_model,test_loader,params,run_mode='test',epoch_i=i)
        # test_result_df.loc[i] = [name_py, test_sensitivity, test_specificity, test_accuracy, test_auc]
        test_result_df.loc[i] = ['aasd', test_sensitivity, test_specificity, test_accuracy, test_auc]


    mean_result_list = ['mean']
    for col in test_result_df.columns[1:]:
        #print(f'col is {col}')
        m = test_result_df[col].mean()
        mean_result_list.append(m)
    test_result_df.loc[test_num] = mean_result_list

    #print(f'show test result, use_scale is {params.use_scale}, use pretrain is {params.use_pretrain} ')
    #print(test_result_df)

    #print(f'run test end')
    return test_result_df


def run_kfold_test(best_pretrain_model_path):
    log.logger.info(f'run kfold test begin, best model path is {best_pretrain_model_path} ')
    best_model = torch.load(best_pretrain_model_path,map_location=torch.device('cpu'))
    #optimizer = torch.optim.Adam(best_model.parameters(),lr=0.001)
    optimizer = torch.optim.Adam(best_model.parameters(),lr=params.lr)
    test_num = CNN_Data.test_num 
    kfold_test_result_df = pd.DataFrame(columns=['patient','Sensitivity','Specificity','Accuracy','AUC']) 
    fold_n = params.fold_n
    log.logger.info(f'run kfold test begin, fold n is {fold_n} ')
    for i in range(fold_n):
        base_model = copy.deepcopy(best_model)
        result_df = run_test_once(base_model)
        kfold_test_result_df = pd.concat([kfold_test_result_df, result_df])
        print(kfold_test_result_df.shape)
    # exit()
    path = '/mnt/data1/user/kenan/result/kfold_result_df.pkl'
    #kfold_test_result_df.to_pickle(path)
    #print(kfold_test_result_df)
    final_result_df = merge_result(kfold_test_result_df)
    print(f'show final result df ')
    print(final_result_df)

def merge_result(kfold_df):
    path = '/mnt/data1/user/kenan/result/kfold_result_df.pkl'
    #kfold_df = pd.read_pickle(path)
    name_list = list(kfold_df['patient'].unique())
    #print(name_list)
    #print(kfold_df)
    #for name in name_list:
    df = kfold_df.groupby(['patient']).mean()

    new_df = pd.DataFrame(columns=['patient','Sensitivity','Specificity','Accuracy','AUC']) 
    for i,index in enumerate(list(df.index)):
        if index == 'mean':
            continue
        new_df.loc[i] = [index] + list(df.loc[index].values)

    #print(new_df)
    #print(df)
    test_num = len(list(new_df.index))
    mean_result_list = ['mean']
    for col in new_df.columns[1:]:
        #print(f'col is {col}')
        m = new_df[col].mean()
        mean_result_list.append(m)
    new_df.loc[test_num] = mean_result_list
    #print(new_df)
    return new_df


if __name__ == '__main__':
    model = cw_srnet()
    # best_pretrain_model_path = 'resnet_scale0_epoch45of50_07301657.pkl'
    # best_pretrain_model_path = 'resnet_scale0_epoch31of50_03301802.pkl'
    best_pretrain_model_path = 'resnet_scale0_epoch45of50_04062047.pkl'
    best_pretrain_model_path = "resnet_scale0_epoch41of50_04190912_谢景恬.pkl"
    run_kfold_test(best_pretrain_model_path)
    exit()
    #optimizer.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)
    #optimizer = torch.optim.Adam(filter(lambda p: 'fc2' not in p.name(), model.parameters()),lr=params.lr)
    module_list = []
    for name, p in model.named_parameters():
        if not ('fc2' in name):
            module_list.append(p)
    optimizer = torch.optim.Adam(module_list, lr=params.lr)
    optimizer2 = torch.optim.Adam(model.fc2.parameters(),lr=params.lr)
    #best_pretrain_model_path = ''
    
    # best_pretrain_model_path = run_train(model, optimizer=optimizer, use_eval=1)
    best_pretrain_model_path = 'resnet_scale0_epoch49of50_06160954.pkl'
    #best_model = torch.load(best_pretrain_model_path)
    run_kfold_test(best_pretrain_model_path)
    log.logger.info(f'-------------- best_pretrain_model_path is : \n {best_pretrain_model_path} ')
    #run_test_once(best_model)
    #run_kfold_test(best_model)
    #merge_result()
    end_time_tag = time.strftime("%m%d%H%M")
    log.logger.info(f'start time tag is {start_time_tag}, end_time_tag is {end_time_tag} ')
    log.logger.info(f'log path is {log_path} ')




