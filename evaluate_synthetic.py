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

