import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn


def pred_item_rank(model_here, test_data):
    data2 = test_data
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 하나밖에 없는 item 제거
    # quantile 이 다 0으로 예측될 것이기 때문
    filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 1].index
    data2 = data2[data2.uid.isin(filter_users)]
    data2 = data2.reset_index()[['uid', 'sid']]
    
    user_num = len(data2.uid.unique())
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()        
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()    
    data2['pred'] = predictions_list
    
    user_item_rank = data2.groupby('uid')['pred'].rank('min', ascending = False)
    data2['user_item_rank'] = user_item_rank.values - 1
    
    user_count = data2.groupby('uid')['pred'].count().reset_index()
    user_count.columns = ['uid', 'user_count']
    user_count['user_count'] = user_count['user_count'] - 1
    user_count_dict = dict(user_count.values)
    
    data2['user_count'] = data2['uid'].map(user_count_dict)
    data2['user_item_rank2'] = data2['user_item_rank'] / data2['user_count']
        
    item_rank = data2[['sid','user_item_rank2']].groupby('sid').mean().reset_index()
    item_rank.columns = ['sid', 'rank']

    return item_rank


def pred_item_score(model_here, test_data):
    
    data = test_data
    
    data['uid'] = data['uid'].apply(lambda x : int(x))
    data['sid'] = data['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
    filter_users = data.uid.value_counts()[data.uid.value_counts() > 1].index
    data = data[data.uid.isin(filter_users)]
    data = data.reset_index()[['uid', 'sid']]
    
    user_num = len(data.uid.unique())
    data_len = data.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data.iloc[frac_user_num*itr:frac_user_num*(itr+1)].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()

        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()
    data['pred'] = predictions_list
    item_score = data[['sid','pred']].groupby('sid').mean().reset_index()

    return item_score



def pred_item_stdscore(model_here, test_data):
    
    data = test_data
    
    data['uid'] = data['uid'].apply(lambda x : int(x))
    data['sid'] = data['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
    # value가 1이면 std deviation이 계산 안되기 때문
    filter_users = data.uid.value_counts()[data.uid.value_counts() > 1].index
    data = data[data.uid.isin(filter_users)]
    data = data.reset_index()[['uid', 'sid']]
    
    user_num = len(data.uid.unique())
    data_len = data.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data.iloc[frac_user_num*itr:frac_user_num*(itr+1)].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()

        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()
    data['pred'] = predictions_list
    user_mean_dict = dict(data.groupby('uid')['pred'].mean().reset_index().values)
    user_std_dict = dict(data.groupby('sid')['pred'].std().reset_index().values)
    
    data['mean'] = data['uid'].map(user_mean_dict)
    data['std'] = data['uid'].map(user_std_dict)
    
    data['z'] = (data['pred'] - data['mean']) / data['std']
    item_z_score = data[['sid','z']].groupby('sid').mean().reset_index()

    return item_z_score



def pred_item_rankdist(model_here, test_data):
    data = test_data
    
    data['uid'] = data['uid'].apply(lambda x : int(x))
    data['sid'] = data['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 4밖에 안되는 user들을 먼저 제거해야함.
    # 4 이하면 quantile 예측이 너무 극단적일 수 있기 때문
    # ex : rate 한 positive item이 하나밖에 없다
    # 이 item의 순위(quantile)은 무조건 0.0 으로 계산됨    
    filter_users = data.uid.value_counts()[data.uid.value_counts() > 4].index
    data = data[data.uid.isin(filter_users)]
    data = data.reset_index()[['uid', 'sid']]
    
    user_num = len(data.uid.unique())
    data_len = data.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data.iloc[frac_user_num*itr:frac_user_num*(itr+1)].values    
        #tmp = tmp.values
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        #label = tmp[:, 2]

        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            #tmp = tmp.values
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()
            #label = tmp[:, 2]

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()
    data['pred'] = predictions_list
    
    user_item_pop_rank = data.groupby('uid')['sid'].rank('min', ascending = True)
    user_item_pop_rank = user_item_pop_rank - 1
    data['user_item_pop_rank'] = user_item_pop_rank
    
    user_item_score_rank = data.groupby('uid')['pred'].rank('min', ascending = False)
    user_item_score_rank = user_item_score_rank
    data['user_item_score_rank'] = user_item_score_rank
    
    user_count = data.groupby('uid')['pred'].count().reset_index()
    user_count.columns = ['uid', 'user_count']
    user_count['user_count'] = user_count['user_count'] - 1
    user_count_dict = dict(user_count.values)
    
    data['user_count'] = data['uid'].map(user_count_dict)
    data['user_item_pop_rank2'] = data['user_item_pop_rank'] / data['user_count']
    data['user_item_score_rank2'] = data['user_item_score_rank'] / data['user_count']
    
    data = data.sort_values(['uid', 'user_item_score_rank2'], ascending = (True, True))
    item_rankdist = data.groupby('uid')['user_item_pop_rank2'].head(1)    
    
    return item_rankdist




def raw_pred_score(model_here, test_data):
    data2 = test_data
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
    filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 1].index
    data2 = data2[data2.uid.isin(filter_users)]
    if 'type' in data2.columns:
        data2 = data2.reset_index()[['uid', 'sid', 'type']]
    else:
        data2 = data2.reset_index()[['uid', 'sid']]
    
    user_num = len(data2.uid.unique())
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        predictions, _ = model_here(user, item, item)
        predictions_list += predictions.detach().cpu().tolist()        
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list += predictions.detach().cpu().tolist()
    
    model_here.cpu()    
    data2['pred'] = predictions_list

    return data2



def uPO(model_here, without_neg_data):
    # https://www.statology.org/pandas-groupby-correlation/
    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.corr.html
    
    # for 문 돌려서 해야되나. 너무 귀찮은데
    data2 = without_neg_data
    filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 3].index
    data2 = data2[data2.uid.isin(filter_users)]
    data2 = data2.reset_index()[['uid', 'sid']]    
    
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
    
    model_here.eval()
    model_here.cuda()
    
    
    tmp = data2.values
    user = tmp[:, 0]
    user = user.astype(np.int32)        
    user = torch.from_numpy(user).cuda()
    item = tmp[:, 1]
    item = item.astype(np.int32)        
    item = torch.from_numpy(item).cuda()
    predictions = model_here.forward_one_item(user, item)
    data2['pred'] = predictions.detach().cpu() 
    
    
    data2 = data2.sort_values(['uid', 'sid'], ascending = [True, False])
    result = data2.groupby('uid')[['sid', 'pred']].corr(method = 'spearman')
    result2 = result.unstack().iloc[:, 1].values.mean()

    
    return result2




# pearson correlation coefficient 계산
def pcc_train(model_here, train_data, sid_pop, item_num):
    data2 = train_data
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
    filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 1].index
    data2 = data2[data2.uid.isin(filter_users)]
    data2 = data2.reset_index()[['uid', 'sid']]
    
    user_num = len(data2.uid.unique())
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = torch.tensor([]).cuda()

    
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        predictions, _ = model_here(user, item, item)
        predictions_list = torch.hstack((predictions_list, predictions))

        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list = torch.hstack((predictions_list, predictions))
            
    
    values = predictions_list.reshape(-1, 1)
    labels = data2.sid.values
    labels = labels.astype(np.int32)
    labels = torch.from_numpy(labels).long().cuda()
    
    M = torch.zeros(item_num, len(values))
    M[labels, torch.arange(len(values))] = 1
    M = torch.nn.functional.normalize(M, p = 1, dim = 1).cuda()
    item_mean_scores = torch.mm(M, values)
    

    sid_pop_labels = sid_pop.sid.values
    sid_pop_labels = sid_pop_labels.astype(np.int32)
    sid_pop_labels = torch.from_numpy(sid_pop_labels).long().cuda()
    sid_pop_values = sid_pop.train_counts.values.reshape(-1,1)
    sid_pop_values = torch.Tensor(sid_pop_values).cuda()
    M = torch.zeros(item_num, len(sid_pop_values))
    M[sid_pop_labels, torch.arange(len(sid_pop_values))] = 1
    M = torch.nn.functional.normalize(M, p = 1, dim = 1).cuda()
    item_pop = torch.mm(M, sid_pop_values)
    
    user_labels = data2.sid.unique()
    user_labels = user_labels.astype(np.int32)
    user_labels = torch.from_numpy(user_labels).long().cuda()
    
    X = item_mean_scores[user_labels]
    Y = item_pop[user_labels]
    pcc = ((X - X.mean())*(Y - Y.mean())).sum() / ((X - X.mean())*(X- X.mean())).sum().sqrt() / ((Y - Y.mean())*(Y- Y.mean())).sum().sqrt()
    #pcc = torch.corrcoef(item_mean_scores[user_labels], item_pop[user_labels])[1,1]
    
    return pcc
    
    
# test 데이터에서 pearson correlation coefficient 계산    
def pcc_test(model_here, test_data, sid_pop, item_num):
    data2 = test_data
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
    filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 1].index
    data2 = data2[data2.uid.isin(filter_users)]
    data2 = data2.reset_index()[['uid', 'sid']]
    
    user_num = len(data2.uid.unique())
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = torch.tensor([]).cuda()
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = user.astype(np.int32)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int32)        
        item = torch.from_numpy(item).cuda()
        predictions, _ = model_here(user, item, item)
        predictions_list = torch.hstack((predictions_list, predictions))
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = user.astype(np.int32)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int32)        
            item = torch.from_numpy(item).cuda()

            predictions, _ = model_here(user, item, item)
            predictions_list = torch.hstack((predictions_list, predictions))
    

    
    values = predictions_list.reshape(-1, 1)
    labels = data2.sid.values
    labels = labels.astype(np.int32)
    labels = torch.from_numpy(labels).long().cuda()
    
    M = torch.zeros(item_num, len(values))
    M[labels, torch.arange(len(values))] = 1
    M = torch.nn.functional.normalize(M, p = 1, dim = 1).cuda()
    item_mean_scores = torch.mm(M, values)
    
    
    sid_pop_labels = sid_pop.sid.values
    sid_pop_labels = sid_pop_labels.astype(np.int32)
    sid_pop_labels = torch.from_numpy(sid_pop_labels).long().cuda()
    sid_pop_values = sid_pop.total_counts.values.reshape(-1,1)
    sid_pop_values = torch.Tensor(sid_pop_values).cuda()
    M = torch.zeros(item_num, len(sid_pop_values))
    M[sid_pop_labels, torch.arange(len(sid_pop_values))] = 1
    M = torch.nn.functional.normalize(M, p = 1, dim = 1).cuda()
    item_pop = torch.mm(M, sid_pop_values)
    
    user_labels = data2.sid.unique()
    user_labels = user_labels.astype(np.int32)
    user_labels = torch.from_numpy(user_labels).long().cuda()
    
    X = item_mean_scores[user_labels]
    Y = item_pop[user_labels]
    pcc = ((X - X.mean())*(Y - Y.mean())).sum() / ((X - X.mean())*(X- X.mean())).sum().sqrt() / ((Y - Y.mean())*(Y- Y.mean())).sum().sqrt()
    
    
    return pcc

