import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

import model_basic
import config
import evaluate
import data_utils

from pop_bias_metrics_basic import pred_item_rank, pred_item_score, pred_item_stdscore, pred_item_rankdist, raw_pred_score, pcc_train, pcc_test, uPO
import scipy.stats as stats
from scipy.stats import skew

import random as random
random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--dropout", 
	type=float,
	default=0.0,  
	help="dropout rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=256, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=15,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=10, 
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=64, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_layers", 
	type=int,
	default=3, 
	help="number of layers in MLP model")
parser.add_argument("--num_ng", 
	type=int,
	default=3, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="1",  
	help="gpu card ID")


parser.add_argument("--dataset", 
	type=str,
	default="movielens",  
	help="movielens, ciao, epinions, gowalla")
parser.add_argument("--model", 
	type=str,
	default='error',  
	help="MF, MLP")
parser.add_argument("--sample", 
	type=str,
	default="none",  
	help="none, pos2, neg2, pos2neg2, pearson")
parser.add_argument("--weight", 
	type=float,
	default=0.5,  
	help="weight")
parser.add_argument("--burnin", 
	type=str,
	default="no",  
	help="yes, no")
parser.add_argument("--reg", 
	type=str,
	default='no',  
	help="yes, no")


args = parser.parse_args()

val_results = []


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True



############################## PREPARE DATASET ##########################

# construct the train and test datasets
user_num, item_num, train_data_len = data_utils.load_all_custom( dataset=args.dataset )

raw_train_data = pd.read_csv(f'./data/final_{args.dataset}/train_df')    
val_data_without_neg = pd.read_csv(f'./data/final_{args.dataset}/val_df')    
val_data_with_neg = pd.read_csv(f'./data/final_{args.dataset}/val_df_with_neg')    
test_data_without_neg = pd.read_csv(f'./data/final_{args.dataset}/test_df')    
test_data_with_neg = pd.read_csv(f'./data/final_{args.dataset}/test_df_with_neg')    
                        
sid_pop_total = pd.read_csv(f'./data/final_{args.dataset}/sid_pop_total')
sid_pop_train = pd.read_csv(f'./data/final_{args.dataset}/sid_pop_train')

train_dataset = data_utils.BPRData(train_data_len*args.num_ng)
train_loader = data.DataLoader(train_dataset,
		batch_size=args.batch_size, shuffle=True, num_workers=4)


########################### CREATE MODEL #################################
GMF_model = None
MLP_model = None



if args.model == 'MLP':
	model = model_basic.BPR(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
if args.model == 'MF':
	model = model_basic.MF_BPR(user_num, item_num, args.factor_num, args.num_layers, 
						args.dropout, config.model, GMF_model, MLP_model)
    

model.cuda()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

########################### TRAINING #####################################


sid_pop_train_dict = dict(list(zip(sid_pop_train.sid, sid_pop_train.train_counts)))

print(args.dataset, ' ', args.model, ' ', args.sample, ' ', args.weight, ' ', 'reg', args.reg, 'burnin', args.burnin)
print('entered training')
count, best_hr = 0, 0

sample = args.sample
acc_w = args.weight
pop_w = 1-args.weight




for epoch in range(args.epochs):
	print('epoch is : ',epoch)        
	model.train() 
	start_time = time.time()
    
	train_loader.dataset.get_data(args.dataset, epoch)
	model.cuda()
    
	if epoch < args.epochs/4:
		if args.burnin == 'yes':
			sample = 'none'            
	elif epoch >= args.epochs/4:
		sample = args.sample        
    
    
	if args.sample in ['none', 'posneg']:
		for user, pos1, pos2, neg1, neg2 in train_loader:    
			pos, neg = pos1, neg1
			_, _ = pos2, neg2                        
            
			user = user.cuda()
			pos = pos.cuda()
			neg = neg.cuda()
			model.zero_grad()
            
			pos_scores, neg_scores = model(user, pos, neg)
			if args.sample == 'none':
				loss = - (pos_scores - neg_scores).sigmoid().log().mean()
			elif args.sample == 'posneg':
				acc_loss = - (pos_scores - neg_scores).sigmoid().log().mean()/2               
				pop_loss =  -(1 -(pos_scores + neg_scores).abs().tanh() ).log().mean()/2
				loss = acc_loss*acc_w + pop_loss*pop_w                
			if args.reg == 'yes':        
				user_emb_w = model.embed_user_MLP.weight[user]        
				pos_emb_w = model.embed_item_MLP.weight[pos]
				neg_emb_w = model.embed_item_MLP.weight[neg]        
				reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos_emb_w) ** 2 + torch.norm(neg_emb_w) ** 2)/3 / args.batch_size
				loss += 1e-5*reg        
			loss.backward()
			optimizer.step()
        
	elif args.sample == 'pos2neg2':
		for user, pos1, pos2, neg1, neg2 in train_loader:
			user = user.cuda()
			pos1 = pos1.cuda()
			pos2 = pos2.cuda()
			neg1 = neg1.cuda()
			neg2 = neg2.cuda()            

			model.zero_grad()
            
			pos1_scores, neg1_scores = model(user, pos1, neg1)            
			pos2_scores, neg2_scores = model(user, pos2, neg2)
            
			acc_loss = - (pos1_scores - neg1_scores).sigmoid().log().mean()/4 - (pos2_scores - neg2_scores).sigmoid().log().mean()/4            
			pop_loss =  -(1 -(pos1_scores - pos2_scores).abs().tanh()).log().mean()/4  -(1 -(neg1_scores - neg2_scores).abs().tanh()).log().mean()/4                                    
			loss = acc_loss*acc_w + pop_loss*pop_w
            
			if args.reg == 'yes':                                
				user_emb_w = model.embed_user_MLP.weight[user]        
				pos1_emb_w = model.embed_item_MLP.weight[pos1]
				pos2_emb_w = model.embed_item_MLP.weight[pos2]            
				neg1_emb_w = model.embed_item_MLP.weight[neg1]        
				neg2_emb_w = model.embed_item_MLP.weight[neg2]               
				reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos1_emb_w) ** 2 + torch.norm(neg1_emb_w) ** 2 + torch.norm(pos2_emb_w) ** 2 + torch.norm(neg2_emb_w) ** 2)/5 / args.batch_size
				loss += 1e-5*reg        
                
			loss.backward()
			optimizer.step()            

	elif sample == 'pos2neg2sum':
		for user, pos1, pos2, neg1, neg2 in train_loader:  
            
			user = user.cuda()
			pos1 = pos1.cuda()
			pos2 = pos2.cuda()
			neg1 = neg1.cuda()
			neg2 = neg2.cuda()            

			model.zero_grad()
            
			pos1_scores, neg1_scores = model(user, pos1, neg1)            
			pos2_scores, neg2_scores = model(user, pos2, neg2)
            
			acc_loss = - (pos1_scores - neg1_scores).sigmoid().log().mean()/4 - (pos2_scores - neg2_scores).sigmoid().log().mean()/4            
			pop_loss =  -(1 -(pos1_scores - pos2_scores).abs().tanh() ).log().mean()/8 -(1 -(neg1_scores - neg2_scores).abs().tanh() ).log().mean()/8  -(1 -(pos1_scores + neg1_scores).abs().tanh() ).log().mean()/8 -(1 -(pos2_scores + neg2_scores).abs().tanh() ).log().mean()/8                 
			loss = acc_loss*acc_w + pop_loss*pop_w
            
			if args.reg == 'yes':                                            
				user_emb_w = model.embed_user_MLP.weight[user]        
				pos1_emb_w = model.embed_item_MLP.weight[pos1]
				pos2_emb_w = model.embed_item_MLP.weight[pos2]            
				neg1_emb_w = model.embed_item_MLP.weight[neg1]        
				neg2_emb_w = model.embed_item_MLP.weight[neg2]               
				reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos1_emb_w) ** 2 + torch.norm(neg1_emb_w) ** 2 + torch.norm(pos2_emb_w) ** 2 + torch.norm(neg2_emb_w) ** 2)/5 / args.batch_size
				loss += 1e-5*reg        
			loss.backward()
			optimizer.step()            
            
        
	elif args.sample == 'pos2':
		for user, pos1, pos2, neg1, neg2 in train_loader:    
			_ = neg2
            
			user = user.cuda()
			pos1 = pos1.cuda()
			pos2 = pos2.cuda()
			neg1 = neg1.cuda()

			model.zero_grad()
            
			pos1_scores, neg1_scores = model(user, pos1, neg1)            
			pos2_scores, neg1_scores = model(user, pos2, neg1)
            
			acc_loss = - (pos1_scores - neg1_scores).sigmoid().log().mean()/4 - (pos2_scores - neg1_scores).sigmoid().log().mean()/4            
			pop_loss =  -(1 -(pos1_scores - pos2_scores).abs().tanh() ).log().mean()/2             
			loss = acc_loss*acc_w + pop_loss*pop_w                     
            
			if args.reg == 'yes':                                                        
				user_emb_w = model.embed_user_MLP.weight[user]        
				pos1_emb_w = model.embed_item_MLP.weight[pos1]
				pos2_emb_w = model.embed_item_MLP.weight[pos2]            
				neg1_emb_w = model.embed_item_MLP.weight[neg1]                       
				reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos1_emb_w) ** 2 + torch.norm(neg1_emb_w) ** 2 + torch.norm(pos2_emb_w) ** 2)/3 / args.batch_size
				loss += 1e-5*reg        
			loss.backward()
			optimizer.step()                    


	elif args.sample == 'neg2':
		for user, pos1, pos2, neg1, neg2 in train_loader:    
			_ = pos2            
            
			user = user.cuda()
			pos1 = pos1.cuda()
			neg1 = neg1.cuda()
			neg2 = neg2.cuda()            

			model.zero_grad()
            
			pos1_scores, neg1_scores = model(user, pos1, neg1)            
			pos1_scores, neg2_scores = model(user, pos1, neg2)
            
			acc_loss = - (pos1_scores - neg1_scores).sigmoid().log().mean()/4 - (pos1_scores - neg2_scores).sigmoid().log().mean()/4            
			pop_loss =  -(1 -(neg1_scores - neg2_scores).abs().tanh() ).log().mean()/2            
			loss = acc_loss*acc_w + pop_loss*pop_w                    
            
			if args.reg == 'yes':                                                                   
				user_emb_w = model.embed_user_MLP.weight[user]        
				pos1_emb_w = model.embed_item_MLP.weight[pos1]
				neg1_emb_w = model.embed_item_MLP.weight[neg1]        
				neg2_emb_w = model.embed_item_MLP.weight[neg2]                       
				reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos1_emb_w) ** 2 + torch.norm(neg1_emb_w) ** 2 + torch.norm(neg2_emb_w) ** 2)/4 / args.batch_size
				loss += 1e-5*reg        
			loss.backward()
			optimizer.step()                  

	elif args.sample == 'pearson':
		for user, pos1, pos2, neg1, neg2 in train_loader:    
			pos, neg = pos1, neg1            
			_, _ = pos2, neg2            

			user = user.cuda()
			pos = pos.cuda()
			neg = neg.cuda()
			model.zero_grad()
            
			pos_scores, neg_scores = model(user, pos, neg)
			loss = - (pos_scores - neg_scores).sigmoid().log().mean()
			if args.reg == 'yes':                                                                   
				user_emb_w = model.embed_user_MLP.weight[user]        
				pos_emb_w = model.embed_item_MLP.weight[pos]
				neg_emb_w = model.embed_item_MLP.weight[neg]        
				reg = (torch.norm(user_emb_w) ** 2 + torch.norm(pos_emb_w) ** 2 + torch.norm(neg_emb_w) ** 2)/3 / args.batch_size
				loss += 1e-5*reg                    
            
			loss.backward()
			optimizer.step()
            
		model.zero_grad()           
		pcc = pcc_train(model, raw_train_data, sid_pop_train, item_num)           
		loss = 100*pcc**2
		loss.backward()
		optimizer.step()        
            

	model.eval()
	print('entered val evaluated')    
	HR, NDCG, ARP = evaluate.metrics_custom_new_bpr(model, val_data_with_neg, args.top_k, sid_pop_total, user_num)
	PCC_TEST = pcc_test(model, val_data_without_neg, sid_pop_total, item_num).detach().cpu()    
    
	score = pred_item_score(model, val_data_without_neg)
	SCC_score_test = stats.spearmanr(score.dropna()['sid'].values, score.dropna()['pred'].values)    
	rank = pred_item_rank(model, val_data_without_neg)    
	SCC_rank_test = stats.spearmanr(rank.dropna()['sid'].values, rank.dropna()['rank'].values)
    
	upo = uPO(model, val_data_without_neg)    
    
	rankdist = pred_item_rankdist(model, val_data_without_neg)
	mean_test = np.mean(rankdist.values)    
	skew_test = skew(rankdist.values)

     
    
	model.eval()    
	elapsed_time = time.time() - start_time
	print("The time elapse of epoch {:03d}".format(epoch) + " is: " + 
			time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
	print("HR: {:.3f}\tNDCG: {:.3f}\tARP: {:.3f}".format(np.mean(HR), np.mean(NDCG), np.mean(ARP)))

	print('PCC_TEST : ', np.round(PCC_TEST, 3))    
	print('SCC_score_test : ', np.round(SCC_score_test[0], 3))        
	print('SCC_rank_test : ', np.round(SCC_rank_test[0], 3))     
	print('upo is :', np.round(upo, 3))            
	print('mean_test : ', np.round(mean_test, 3))        
	print('skew_test : ', np.round(skew_test, 3))        
	print(' ')    
	epoch_val_result = [args.batch_size, epoch, args.sample, args.weight, HR, NDCG, ARP, PCC_TEST.numpy(), SCC_score_test[0], SCC_rank_test[0], np.round(upo, 3), mean_test, skew_test]
	val_results.append(epoch_val_result)    

	if HR > best_hr:
		best_hr, best_ndcg, best_arp, best_epoch = HR, NDCG, ARP, epoch    
	if args.out:
		if not os.path.exists(config.model_path):
			os.mkdir(config.model_path)
		torch.save(model, 
			'{}{}_{}_{}.pth'.format(config.model_path,f'final_{args.dataset}_', f'{args.model}_{args.sample}', args.epochs))

        

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}, ARP = {:.3f}".format(
									best_epoch, best_hr, best_ndcg, best_arp))


model.cuda()
model.eval()
print(' ')
print('entered test evaluated')    
'HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)'
HR, NDCG, ARP = evaluate.metrics_custom_new_bpr(model, test_data_with_neg, args.top_k, sid_pop_total, user_num)
PCC_TEST = pcc_test(model, test_data_without_neg, sid_pop_total, item_num).detach().cpu()    
    
score = pred_item_score(model, test_data_without_neg)
SCC_score_test = stats.spearmanr(score.dropna()['sid'].values, score.dropna()['pred'].values)    
rank = pred_item_rank(model, test_data_without_neg)    
SCC_rank_test = stats.spearmanr(rank.dropna()['sid'].values, rank.dropna()['rank'].values)

upo = uPO(model, test_data_without_neg)

rankdist = pred_item_rankdist(model, test_data_without_neg)
mean_test = np.mean(rankdist.values)    
skew_test = skew(rankdist.values)

epoch_val_result = [args.batch_size, -1, args.sample, args.weight, HR, NDCG, ARP, PCC_TEST.numpy(), SCC_score_test[0], SCC_rank_test[0], np.round(upo,3), mean_test, skew_test]
val_results.append(epoch_val_result)

experiment_results = pd.DataFrame(val_results)
experiment_results.columns = ['batch', 'epoch', 'sample', 'weight', 'HR', 'NDCG', 'ARP', 'PCC', 'SCC_score', 'SCC_rank', 'upo', 'mean', 'skew']

experiment_results.to_csv('{}{}_{}_{}_{}_{}_burnin{}_reg{}.csv'.format('./experiments/',args.model, args.dataset, args.sample, args.epochs, np.round(args.weight, 2), args.burnin, args.reg))

    
elapsed_time = time.time() - start_time
print(args.dataset, ' ', args.model, ' ', args.sample, ' ', args.weight, ' ', 'reg', args.reg, 'burnin', args.burnin)
print(' ')
print("HR: {:.3f}\tNDCG: {:.3f}\tARP: {:.3f}".format(np.mean(HR), np.mean(NDCG), np.mean(ARP)))
print('PCC_TEST : ', np.round(PCC_TEST, 3))    
print('SCC_score_test : ', np.round(SCC_score_test[0], 3))        
print('SCC_rank_test : ', np.round(SCC_rank_test[0], 3))     
print('upo is :', np.round(upo, 3)) 
print('mean_test : ', np.round(mean_test, 3))        
print('skew_test : ', np.round(skew_test, 3))        
print(' ')    
    





