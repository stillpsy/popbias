import numpy as np
import torch
import pandas as pd



def calc_acc(pred_mtx):    
    import pandas as pd
    
    if pred_mtx.shape[0] == 200:
        
        user_item_mtx = np.zeros((200, 200))
        for i in range(200):
            for j in range(200):
                if (i < j):
                    user_item_mtx[i,200-j-1] = 1            
                if (i > j):
                    user_item_mtx[i,200-j-1] = 0
        user_item_mtx[200-1, 0] = 1
        
        full_data = pd.DataFrame(user_item_mtx).unstack().reset_index()
        full_data.columns = ['sid', 'uid', 'type']    
        pred_scores = pd.DataFrame(pred_mtx).unstack().reset_index()
        pred_scores.columns = ['sid', 'uid', 'pred']
        
        tmp = full_data.merge(pred_scores, on = ['uid', 'sid'])
        pos_pred = tmp[tmp.type == 1.0]
        neg_pred = tmp[tmp.type == 0.0]
        
        pos_pred = pos_pred[['uid', 'sid', 'pred']].reset_index()[['uid', 'sid', 'pred']]
        neg_pred = neg_pred[['uid', 'sid', 'pred']].reset_index()[['uid', 'sid', 'pred']]
        acc_df = pos_pred.merge(neg_pred, on = 'uid')
        acc_df.columns = ['uid', 'pos_item', 'pos_score', 'neg_item', 'neg_score']
        accuracy = (acc_df['pos_score'] > acc_df['neg_score']).mean()
        return accuracy
        
    elif pred_mtx.shape[0] == 400:
        user_item_mtx = np.zeros((200, 200))
        for i in range(200):
            for j in range(200):
                if (i < j):
                    user_item_mtx[i,200-j-1] = 1            
                if (i > j):
                    user_item_mtx[i,200-j-1] = 0
        user_item_mtx2 = np.ones((200, 200))
        user_item_mtx2[:, 199] = 0
        user_item_mtx = np.concatenate([user_item_mtx, user_item_mtx2], axis = 0)
        
        full_data = pd.DataFrame(user_item_mtx).unstack().reset_index()
        full_data.columns = ['sid', 'uid', 'type']
    
        pred_scores = pd.DataFrame(pred_mtx).unstack().reset_index()
        pred_scores.columns = ['sid', 'uid', 'pred']
        
        tmp = full_data.merge(pred_scores, on = ['uid', 'sid'])
        pos_pred = tmp[tmp.type == 1.0]
        neg_pred = tmp[tmp.type == 0.0]
        
        pos_pred = pos_pred[['uid', 'sid', 'pred']].reset_index()[['uid', 'sid', 'pred']]
        neg_pred = neg_pred[['uid', 'sid', 'pred']].reset_index()[['uid', 'sid', 'pred']]
        
        acc_df = pos_pred.merge(neg_pred, on = 'uid')
        acc_df.columns = ['uid', 'pos_item', 'pos_score', 'neg_item', 'neg_score']
        accuracy = (acc_df['pos_score'] > acc_df['neg_score']).mean()
        return accuracy
        



def metrics_custom_new_bpr(model):
    
    model.eval()
    model.cuda()
    user_emb = model.embed_user_MLP.weight.detach().cpu()
    item_emb = model.embed_item_MLP.weight.detach().cpu()
    user_num = user_emb.shape[0]
    item_num = item_emb.shape[0]
    pred_mtx = np.zeros((user_num, item_num))

    for user in range(user_num):
        pos_score, _ = model(torch.tensor([user]*item_num).cuda(), torch.tensor(list(range(item_num))).cuda(), torch.tensor(list(range(item_num))).cuda())
        pred_mtx[user,:] = pos_score.cpu().detach()

    accuracy = calc_acc(pred_mtx)
    
    return accuracy



def metrics_graph_bpr(model):
    model.eval()
    model.cuda()

    item_emb = model.embedding_dict.item_emb.detach().cpu()
    user_emb = model.embedding_dict.user_emb.detach().cpu()
    user_num = user_emb.shape[0]
    item_num = item_emb.shape[0]
    pred_mtx = np.zeros((user_num, item_num))
    for user in range(user_num):
        u_emb, pos_emb, _ = model(torch.tensor([user]*item_num).cuda(), torch.tensor(list(range(item_num))).cuda(), torch.tensor(list(range(item_num))).cuda(), drop_flag = False)
        pred_mtx[user,:] = torch.sum(torch.mul(u_emb, pos_emb), axis=1).cpu().detach()
    
    accuracy = calc_acc(pred_mtx)

    return accuracy


'''

def arp_custom_new(pred_items, sid_pop_total):
	#pred_items # 6040 x top_k
	#Error code : mapped_pred_items = list(map(sid_pop_total.get, pred_items))
	#Error code : mapped_pred_items = list(map(sid_pop_total.get, pred_items))
	d = sid_pop_total
	l = pred_items
    
	r = [ [ d[v] for v in lv] for lv in l]
	ARP = np.mean(np.sum(r, axis = 1))
	return ARP



# New code - 20220304
# For Small Data
def metrics_custom_new_bpr(model, test_data, top_k, sid_pop_total, user_num):
	top_k = 3    
	import pandas as pd
	import random    
	HR, NDCG, ARP = [], [], []

	test_data = test_data[['uid', 'sid', 'type']]
	test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
	test_data['sid'] = test_data['sid'].apply(lambda x : int(x))    
	test_users_num = len(test_data['uid'].unique())    

	# 여기가 한번에 안 돌아가서 문제 같음., 특히 MLP 모델에서    
	user = test_data.values[:, 0]
	user = user.astype(np.int32)        
	user = torch.from_numpy(user).cuda()
	item = test_data.values[:, 1]
	item = item.astype(np.int32)
	item = torch.from_numpy(item).cuda()                          
	predictions = model.forward_one_item(user, item)                              
	test_data['pred'] = predictions.detach().cpu()
    
	data_len = test_data.shape[0]
	frac = 50
	frac_user_num = int(data_len/frac)
	predictions_list = []
	model.eval()
	model.cuda()
    
	predictions_list = [] 
    
	for itr in range(frac):
		tmp = test_data.iloc[ (frac_user_num * itr) : (frac_user_num* (itr+1) ) ].values        
		user = tmp[:, 0]
		user = user.astype(np.int32)
		user = torch.from_numpy(user).cuda()
		item = tmp[:, 1]
		item = item.astype(np.int32)        
		item = torch.from_numpy(item).cuda()
		predictions_tmp = model.forward_one_item(user, item)        
		predictions_list += predictions_tmp.detach().cpu().tolist()
	if itr+1 == frac:
		tmp = test_data.iloc[ (frac_user_num * (itr+1)):].values        
		user = tmp[:, 0]
		user = user.astype(np.int32)
		user = torch.from_numpy(user).cuda()
		item = tmp[:, 1]
		item = item.astype(np.int32)        
		item = torch.from_numpy(item).cuda()
		predictions_tmp = model.forward_one_item(user, item)        
		predictions_list += predictions_tmp.detach().cpu().tolist()
        
	test_data['pred'] = predictions_list        
    
	### compute ARP    
	pred_result_for_arp = test_data.sort_values(['pred'], ascending = False).groupby('uid').head(top_k)
	pred_result_for_arp = pred_result_for_arp.sid.astype(int).values
	ARP.append(pred_result_for_arp.mean())

	### compute accuracy, ndcg @ top k
	test_data = test_data[~test_data.uid.isin([0, 1, 2, 3])]       
	test_data_pos = test_data[test_data.type == 'pos']
	test_data_pos = test_data_pos.reset_index()[['uid', 'sid', 'pred']]

	test_data_neg = test_data[test_data.type == 'neg']
	test_data_neg = test_data_neg.reset_index()[['uid', 'sid', 'pred']]
	test_data_neg = test_data_neg.sort_values(by = ['uid', 'pred'], ascending = [True, False])
	test_data_neg['order'] = list(range(top_k)) * (len(test_data_neg.uid.unique()))
	neg_scores_table_alluser = test_data_neg.pivot_table(index = 'uid', columns = 'order', values = 'pred')   # user_num x 100
	neg_scores_table_testuser = neg_scores_table_alluser.loc[test_data_pos.uid.values, :]       # test_pos_item_num x 100
    
	pos_score_final = pd.DataFrame(test_data_pos.pred.values) # test pos item num x 1
	neg_score_final = pd.DataFrame(neg_scores_table_testuser.values) # test pos item num x 100
    
	final_df = pd.concat([pos_score_final, neg_score_final], axis = 1) # test_pos_item_num x 101
	final_df.columns = list(range(top_k+1))        
	rank_list = final_df.iloc[:, 0:(top_k+1)]    
	rank_score = rank_list.rank(1, ascending=False, method='max').iloc[:,0].values
	hits = (rank_score < 2)*1
	ndcgs = hits*np.reciprocal(np.log2(rank_score+1))    
    
	HR += hits.tolist()
	NDCG += ndcgs.tolist()

	return np.mean(HR), np.mean(NDCG), np.mean(ARP)






def metrics_graph_bpr(model, test_data, top_k, sid_pop_total, user_num):
	top_k = 3    
    
	import pandas as pd
	import random    
	HR, NDCG, ARP = [], [], []

	test_data = test_data[['uid', 'sid', 'type']]
	test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
	test_data['sid'] = test_data['sid'].apply(lambda x : int(x))    
	test_users_num = len(test_data['uid'].unique())    

	user = test_data.values[:, 0]
	user = np.array(user).astype(np.int32)
	user = user.tolist()
	user = torch.LongTensor(user).cuda()
    
	item = test_data.values[:, 1] #.tolist()
	item = np.array(item).astype(np.int32)        
	item = item.tolist()    
	item = torch.LongTensor(item).cuda()

	u_emb, pos_i_emb, neg_i_emb = model(user, item, item, drop_flag = False)
	predictions = torch.sum(torch.mul(u_emb, pos_i_emb), axis=1)    
	test_data['pred'] = predictions.detach().cpu()
    
	### compute ARP
	pred_result_for_arp = test_data.sort_values(['pred'], ascending = False).groupby('uid').head(top_k)
	pred_result_for_arp = pred_result_for_arp.sid.astype(int).values
	ARP.append(pred_result_for_arp.mean())

	### compute accuracy, ndcg @ top k
	test_data = test_data[~test_data.uid.isin([0, 1, 2, 3])]   
	test_data_pos = test_data[test_data.type == 'pos']
	test_data_pos = test_data_pos.reset_index()[['uid', 'sid', 'pred']]

	test_data_neg = test_data[test_data.type == 'neg']
	test_data_neg = test_data_neg.reset_index()[['uid', 'sid', 'pred']]
	test_data_neg = test_data_neg.sort_values(by = ['uid', 'pred'], ascending = [True, False])
	test_data_neg['order'] = list(range(top_k)) * (len(test_data_neg.uid.unique()))
	neg_scores_table_alluser = test_data_neg.pivot_table(index = 'uid', columns = 'order', values = 'pred')   # user_num x 100
	neg_scores_table_testuser = neg_scores_table_alluser.loc[test_data_pos.uid.values, :]       # test_pos_item_num x 100
    
	pos_score_final = pd.DataFrame(test_data_pos.pred.values) # test pos item num x 1
	neg_score_final = pd.DataFrame(neg_scores_table_testuser.values) # test pos item num x 100
    
	final_df = pd.concat([pos_score_final, neg_score_final], axis = 1) # test_pos_item_num x 101
	final_df.columns = list(range(top_k + 1))        
	rank_list = final_df.iloc[:, 0:(top_k + 1)]
	rank_score = rank_list.rank(1, ascending=False, method='max').iloc[:,0].values
	hits = (rank_score < 2)*1
	ndcgs = hits*np.reciprocal(np.log2(rank_score+1))    

	HR += hits.tolist()
	NDCG += ndcgs.tolist()

	return np.mean(HR), np.mean(NDCG), np.mean(ARP)
   
'''