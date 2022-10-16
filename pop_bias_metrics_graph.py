import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn


def pred_item_rank(model_here, test_data, sid_pop_total):
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
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list += predictions.detach().cpu().tolist()        
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list += predictions.detach().cpu().tolist()        
    
    model_here.cpu()    
    data2['pred'] = predictions_list
    
    user_item_rank = data2.groupby('uid')['pred'].rank('average', ascending = False)
    data2['user_item_rank'] = user_item_rank.values - 1
    
    user_count = data2.groupby('uid')['pred'].count().reset_index()
    user_count.columns = ['uid', 'user_count']
    user_count['user_count'] = user_count['user_count'] - 1
    user_count_dict = dict(user_count.values)
    
    data2['user_count'] = data2['uid'].map(user_count_dict)
    data2['user_item_rank2'] = data2['user_item_rank'] / data2['user_count']
        
    item_rank = data2[['sid','user_item_rank2']].groupby('sid').mean().reset_index()
    item_rank.columns = ['sid', 'rank']
    
    sid_pop_dict = dict(sid_pop_total.values)
    item_rank['sid_pop_count'] = item_rank['sid'].map(sid_pop_dict)    

    return item_rank


def pred_item_score(model_here, test_data, sid_pop_total):
    
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
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list += predictions.detach().cpu().tolist()        
        
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list += predictions.detach().cpu().tolist()        
            
    model_here.cpu()
    data['pred'] = predictions_list
    item_score = data[['sid','pred']].groupby('sid').mean().reset_index()

    sid_pop_dict = dict(sid_pop_total.values)
    item_score['sid_pop_count'] = item_score['sid'].map(sid_pop_dict)

    return item_score



def pred_item_stdscore(model_here, test_data):
    
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
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list += predictions.detach().cpu().tolist()        
                
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
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



def pred_item_rankdist(model_here, test_data, sid_pop_total):
    data = test_data
    
    data['uid'] = data['uid'].apply(lambda x : int(x))
    data['sid'] = data['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
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
        user = tmp[:, 0]
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list += predictions.detach().cpu().tolist()        
        
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list += predictions.detach().cpu().tolist()        

    
    model_here.cpu()
    data['pred'] = predictions_list
    
    sid_pop_dict = dict(sid_pop_total.values)
    data['sid_pop_count'] = data['sid'].map(sid_pop_dict)    
    
    user_item_pop_rank = data.groupby('uid')['sid_pop_count'].rank('average', ascending = False)
    data['user_item_pop_rank'] = user_item_pop_rank -1
    
    user_item_score_rank = data.groupby('uid')['pred'].rank('average', ascending = False)
    data['user_item_score_rank'] = user_item_score_rank -1
    
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

def pred_item_rankdist2(model_here, test_data, sid_pop_total):
    data = test_data
    
    data['uid'] = data['uid'].apply(lambda x : int(x))
    data['sid'] = data['sid'].apply(lambda x : int(x))    
        
    # 먼저 value가 1밖에 안되는 user들을 먼저 제거해야함.
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
        user = tmp[:, 0]
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list += predictions.detach().cpu().tolist()        
        
        
        if itr+1 == frac:
            tmp = data.iloc[frac_user_num*(itr+1):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list += predictions.detach().cpu().tolist()        

    
    model_here.cpu()
    data['pred'] = predictions_list
    
    sid_pop_dict = dict(sid_pop_total.values)
    data['sid_pop_count'] = data['sid'].map(sid_pop_dict)    
    
    user_item_pop_rank = data.groupby('uid')['sid_pop_count'].rank('average', ascending = False)
    data['user_item_pop_rank'] = user_item_pop_rank -1
    
    user_item_score_rank = data.groupby('uid')['pred'].rank('average', ascending = False)
    data['user_item_score_rank'] = user_item_score_rank -1
    
    user_count = data.groupby('uid')['pred'].count().reset_index()
    user_count.columns = ['uid', 'user_count']
    user_count['user_count'] = user_count['user_count'] - 1
    user_count_dict = dict(user_count.values)
    
    data['user_count'] = data['uid'].map(user_count_dict)
    data['user_item_pop_rank2'] = data['user_item_pop_rank'] / data['user_count']
    data['user_item_score_rank2'] = data['user_item_score_rank'] / data['user_count']
    
    data = data.sort_values(['uid', 'user_item_score_rank2'], ascending = (True, True))

    res = data[['user_item_pop_rank2', 'user_item_score_rank2']]
    res.columns = ['pop_rank', 'score_rank']

    
    bins = np.linspace(0, 1, 20)
    
    res['bins'] = pd.cut(res['pop_rank'], bins=bins, include_lowest=True)    
    
    
    return res


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
    frac = 100
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list += predictions.detach().cpu().tolist()        
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list += predictions.detach().cpu().tolist()        
    
    model_here.cpu()    
    data2['pred'] = predictions_list

    return data2


def uPO(model_here, without_neg_data, sid_pop_total):
    # https://www.statology.org/pandas-groupby-correlation/
    # https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.DataFrameGroupBy.corr.html
    
    data2 = without_neg_data
    filter_users = data2.uid.value_counts()[data2.uid.value_counts() > 3].index
    data2 = data2[data2.uid.isin(filter_users)]
    data2 = data2.reset_index()[['uid', 'sid']]    
    
    
    data2['uid'] = data2['uid'].apply(lambda x : int(x))
    data2['sid'] = data2['sid'].apply(lambda x : int(x))    
    
    model_here.eval()
    model_here.cuda()
    
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = []
    
    model_here.eval()
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list += predictions.detach().cpu().tolist()        
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list += predictions.detach().cpu().tolist()        
    
    model_here.cpu()
    data2['pred'] = predictions_list    
    
    data2 = data2.sort_values(['uid', 'sid'], ascending = [True, False])    
    sid_pop_dict = dict(sid_pop_total.values)
    data2['sid_pop_count'] = data2['sid'].map(sid_pop_dict)
    result = data2.groupby('uid')[['sid_pop_count', 'pred']].corr(method = 'spearman')
    result2 = result.unstack().iloc[:, 1].fillna(0).values.mean()

    
    return result2
    



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
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list = torch.hstack((predictions_list, predictions))
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list = torch.hstack((predictions_list, predictions))

    sid_pop_dict = dict(sid_pop.values)
    data2['sid_pop_count'] = data2['sid'].map(sid_pop_dict)            
        
    values = predictions_list.reshape(-1, 1)
    sid_pop_count = data2.sid_pop_count.values
    sid_pop_count = sid_pop_count.astype(np.int32)
    sid_pop_count = torch.from_numpy(sid_pop_count).float().cuda()
    
    X = values
    Y = sid_pop_count # item pop
    
    pcc = torch.corrcoef([X, Y])[0, 1]
    
    #pcc = ((X - X.mean())*(Y - Y.mean())).sum() / ((X - X.mean())*(X- X.mean())).sum().sqrt() / ((Y - Y.mean())*(Y- Y.mean())).sum().sqrt()    

    
    return pcc
    
    
    
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
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list = torch.hstack((predictions_list, predictions))
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list = torch.hstack((predictions_list, predictions))
    
    sid_pop_dict = dict(sid_pop.values)
    data2['sid_pop_count'] = data2['sid'].map(sid_pop_dict)            
        
    data2['pred'] = predictions_list.detach().cpu().tolist()
    values = data2.pred.values
    sid_pop_count = data2.sid_pop_count.values
    
    
    X = np.array(values)
    Y = np.array(sid_pop_count) # item pop
    
    pcc = ((X - X.mean())*(Y - Y.mean())).sum() / np.sqrt(((X - X.mean())*(X- X.mean())).sum()) / np.sqrt(((Y - Y.mean())*(Y- Y.mean())).sum())    

    
    return pcc


def pcc_test_check(model_here, test_data, sid_pop_total):
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
        user = np.array(user).astype(np.int32)
        user = user.tolist()
        user = torch.LongTensor(user).cuda()

        item = tmp[:, 1]
        item = np.array(item).astype(np.int32)        
        item = item.tolist()    
        item = torch.LongTensor(item).cuda()

        u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
        predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
        predictions_list = torch.hstack((predictions_list, predictions))
        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = np.array(user).astype(np.int32)
            user = user.tolist()
            user = torch.LongTensor(user).cuda()

            item = tmp[:, 1]
            item = np.array(item).astype(np.int32)        
            item = item.tolist()    
            item = torch.LongTensor(item).cuda()

            u_emb, pos_i_emb, neg_i_emb = model_here(user, item, item, drop_flag = False)
            predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)
            predictions_list = torch.hstack((predictions_list, predictions))

    data2['pred'] = predictions_list.detach().cpu().tolist()
            
    sid_pop_dict = dict(sid_pop.values)
    data2['sid_pop_count'] = data2['sid'].map(sid_pop_dict)            
        
    result = data2[['sid_pop_count', 'pred']].corr(method = 'pearson')

    return result
