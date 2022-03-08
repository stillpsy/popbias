import numpy as np
import torch
import pandas as pd



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
    
	import pandas as pd
	import random    
	HR, NDCG, ARP = [], [], []

	test_data = test_data[['uid', 'sid', 'type']]
	test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
	test_data['sid'] = test_data['sid'].apply(lambda x : int(x))    
	test_users_num = len(test_data['uid'].unique())    

	user = test_data.values[:, 0]
	user = user.astype(np.int32)        
	user = torch.from_numpy(user).cuda()
	item = test_data.values[:, 1]
	item = item.astype(np.int32)
	item = torch.from_numpy(item).cuda()                          
	predictions, _ = model(user, item, item)                          
	test_data['pred'] = predictions.detach().cpu()
    
	### compute ARP
	pred_result_for_arp = test_data.sort_values(['pred'], ascending = False).groupby('uid').head(top_k)
	pred_result_for_arp = pred_result_for_arp.sid.astype(int).values
	pred_result_for_arp = pred_result_for_arp.reshape(test_users_num, top_k)
	pop_count_dict = dict(zip(sid_pop_total.sid, sid_pop_total.total_counts))
	ARP.append(arp_custom_new(pred_result_for_arp.tolist(), pop_count_dict))

	### compute accuracy, ndcg @ top k
	test_data_pos = test_data[test_data.type == 'pos']
	test_data_pos = test_data_pos.reset_index()[['uid', 'sid', 'pred']]

	test_data_neg = test_data[test_data.type == 'neg']
	test_data_neg = test_data_neg.reset_index()[['uid', 'sid', 'pred']]
	test_data_neg = test_data_neg.sort_values(by = ['uid', 'pred'], ascending = [True, False])
	test_data_neg['order'] = list(range(100)) * user_num
	neg_scores_table_alluser = test_data_neg.pivot_table(index = 'uid', columns = 'order', values = 'pred')   # user_num x 100
	neg_scores_table_testuser = neg_scores_table_alluser.loc[test_data_pos.uid.values, :]       # test_pos_item_num x 100
    
	pos_score_final = pd.DataFrame(test_data_pos.pred.values) # test pos item num x 1
	neg_score_final = pd.DataFrame(neg_scores_table_testuser.values) # test pos item num x 100
    
	final_df = pd.concat([pos_score_final, neg_score_final], axis = 1) # test_pos_item_num x 101
	final_df.columns = list(range(101))        
	rank_list = final_df.iloc[:, 0:(top_k+1)]
	rank_score = rank_list.rank(1, ascending=False, method='average').iloc[:,0].values
	hits = (rank_score < (top_k + 1))*1
	ndcgs = hits*np.reciprocal(np.log2(rank_score+1))    

	HR += hits.tolist()
	NDCG += ndcgs.tolist()

	return np.mean(HR), np.mean(NDCG), np.mean(ARP)






def metrics_graph_bpr(model, test_data, top_k, sid_pop_total, user_num):
    
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
	pred_result_for_arp = pred_result_for_arp.reshape(test_users_num, top_k)
	pop_count_dict = dict(zip(sid_pop_total.sid, sid_pop_total.total_counts))
	ARP.append(arp_custom_new(pred_result_for_arp.tolist(), pop_count_dict))

	### compute accuracy, ndcg @ top k
	test_data_pos = test_data[test_data.type == 'pos']
	test_data_pos = test_data_pos.reset_index()[['uid', 'sid', 'pred']]

	test_data_neg = test_data[test_data.type == 'neg']
	test_data_neg = test_data_neg.reset_index()[['uid', 'sid', 'pred']]
	test_data_neg = test_data_neg.sort_values(by = ['uid', 'pred'], ascending = [True, False])
	test_data_neg['order'] = list(range(100)) * user_num
	neg_scores_table_alluser = test_data_neg.pivot_table(index = 'uid', columns = 'order', values = 'pred')   # user_num x 100
	neg_scores_table_testuser = neg_scores_table_alluser.loc[test_data_pos.uid.values, :]       # test_pos_item_num x 100
    
	pos_score_final = pd.DataFrame(test_data_pos.pred.values) # test pos item num x 1
	neg_score_final = pd.DataFrame(neg_scores_table_testuser.values) # test pos item num x 100
    
	final_df = pd.concat([pos_score_final, neg_score_final], axis = 1) # test_pos_item_num x 101
	final_df.columns = list(range(101))        
	rank_list = final_df.iloc[:, 0:(top_k+1)]
	rank_score = rank_list.rank(1, ascending=False, method='average').iloc[:,0].values
	hits = (rank_score < (top_k + 1))*1
	ndcgs = hits*np.reciprocal(np.log2(rank_score+1))    

	HR += hits.tolist()
	NDCG += ndcgs.tolist()

	return np.mean(HR), np.mean(NDCG), np.mean(ARP)
