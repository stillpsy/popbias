import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data
import config
import random as random

'''
def load_all_custom(test_num=100, dataset=None):
	""" We load all the three file here to save time in each epoch. """
    
	total_data = pd.read_csv(f'./data/final_{dataset}/total_df')    
	total_data = total_data[['uid', 'sid']]    
	total_data['uid'] = total_data['uid'].apply(lambda x : int(x))
	total_data['sid'] = total_data['sid'].apply(lambda x : int(x))    
	user_num = total_data['uid'].max() + 1
	item_num = total_data['sid'].max() + 1
	del total_data    
    
	train_data = pd.read_csv(f'./data/final_{dataset}/train_df')    
	train_data = train_data[['uid', 'sid']]
	train_data['uid'] = train_data['uid'].apply(lambda x : int(x))
	train_data['sid'] = train_data['sid'].apply(lambda x : int(x))    
	#train_data.columns = ['user', 'item']
	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		train_mat[x[0], x[1]] = 1.0

	#test_data = pd.read_csv('./data/test_df_true_neg')        
	#test_data = pd.read_csv('./data/synthetic/test_df_true_neg')
	test_data = pd.read_csv(f'./data/final_{dataset}/test_df')     
	test_data = test_data[['uid', 'sid']]
	test_data['uid'] = test_data['uid'].apply(lambda x : int(x))
	test_data['sid'] = test_data['sid'].apply(lambda x : int(x))
	test_data.columns = ['user', 'item']    
	test_data = test_data.values.tolist()    
    
	val_data = pd.read_csv(f'./data/final_{dataset}/val_df')     
	val_data = val_data[['uid', 'sid']]
	val_data['uid'] = val_data['uid'].apply(lambda x : int(x))
	val_data['sid'] = val_data['sid'].apply(lambda x : int(x))
	val_data.columns = ['user', 'item']    
	val_data = val_data.values.tolist()        

	neg_samples_data = pd.read_csv(f'./data/final_{dataset}/neg_sample_df')     
	neg_samples_data = neg_samples_data[['uid', 'sid']]
	neg_samples_data['uid'] = neg_samples_data['uid'].apply(lambda x : int(x))
	neg_samples_data['sid'] = neg_samples_data['sid'].apply(lambda x : int(x))
	neg_samples_data.columns = ['user', 'item']    
	neg_samples_data = neg_samples_data.values.tolist()            

        
	total_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	for x in train_data:
		total_mat[x[0], x[1]] = 1.0
	for x in test_data:
		total_mat[x[0], x[1]] = 1.0
	for x in val_data:
		total_mat[x[0], x[1]] = 1.0
	for x in neg_samples_data:
		total_mat[x[0], x[1]] = 1.0
        
	test_data = None # dummy code
	test_mat = None  # dummy code

	return train_data, test_data, user_num, item_num, train_mat, test_mat, total_mat
'''

def load_all_custom(test_num=100, dataset=None):
	""" We load all the three file here to save time in each epoch. """
    
	total_data = pd.read_csv(f'./data/final_{dataset}/total_df')    
	total_data = total_data[['uid', 'sid']]    
	total_data['uid'] = total_data['uid'].apply(lambda x : int(x))
	total_data['sid'] = total_data['sid'].apply(lambda x : int(x))    
	user_num = total_data['uid'].max() + 1
	item_num = total_data['sid'].max() + 1
    
	train_data = pd.read_csv(f'./data/final_{dataset}/train_df')    
	train_data = train_data[['uid', 'sid']]
	train_data_len = train_data.shape[0]
    
	return user_num, item_num, train_data_len




'''
class BPRData(data.Dataset):
	def __init__(self, features, 
				num_item, train_mat=None, total_mat=None, num_ng=0, is_training=None, sample_mode = None):
		super(BPRData, self).__init__()
		""" Note that the labels are only useful when training, we thus 
			add them in the ng_sample() function.
		"""
		self.features = features
		self.features2 = None
		self.num_item = num_item
		self.train_mat = train_mat
		self.total_mat = total_mat
		self.num_ng = num_ng
		self.is_training = is_training        
		self.sample_mode = sample_mode        
		# self.labels = [0 for _ in range(len(features))]

	def ng_sample(self):
		### sample 2 pos, 2 neg        

		if True:
			assert self.is_training, 'no need to sampling when testing'
			self.features_fill = []
			### self.features is train [user, pos item] list
			tmp = pd.DataFrame(self.features)
			tmp.columns = ['uid', 'sid']
            
			### [user pos] -> [user pos1 pos2] 
			### by groupby uid, then shuffling sid
			tmp = tmp.sort_values('uid')
			tmp_list = list(range(tmp.shape[0]))
			random.shuffle(tmp_list)
			tmp['rng'] = tmp_list
			sid2 = tmp.sort_values(['uid', 'rng']).sid
			tmp['sid2'] = sid2.reset_index().sid
			tmp = tmp[['uid', 'sid', 'sid2']]
			tmp = tmp.sort_index()
			self.features2 = tmp.values.tolist()   
            
			### add neg1, neg2
			### random sample until neg1, neg2 is not from total_mat            
			### note total_mat includes train, val, test, (test neg_samples)            
			for x in self.features2:
				u, pos1, pos2 = x[0], x[1], x[2]
				for t in range(self.num_ng):
					neg1, neg2 = np.random.randint(self.num_item, size = 2)
					while ((u, neg1) in self.total_mat) or ((u, neg2) in self.total_mat):
						neg1, neg2 = np.random.randint(self.num_item, size = 2)
					self.features_fill.append([u, pos1, pos2, neg1, neg2])
            

	def __len__(self):
		return self.num_ng * len(self.features) if self.is_training \
					else len(self.features)

	def __getitem__(self, idx):
		features = self.features_fill if \
					self.is_training else self.features
        
		if True:    
			user = features[idx][0]
			pos1 = features[idx][1]
			pos2 = features[idx][2]        
			neg1 = features[idx][3]                    
			neg2 = features[idx][4]                                
			return user, pos1, pos2, neg1, neg2
'''




class BPRData(data.Dataset):
	def __init__(self, train_data_length):
		super(BPRData, self).__init__()
		self.train_data_length = train_data_length
		self.features_fill = None        

	def get_data(self, dataset, current_epoch):
		import pickle
		with open(f'./data/final_{dataset}/train_samples_{current_epoch}', 'rb') as fp:
			b = pickle.load(fp)
			self.features_fill = b            
            
	def __len__(self):
		return self.train_data_length

	def __getitem__(self, idx):
		features = self.features_fill 
		if True:    
			user = features[idx][0]
			pos1 = features[idx][1]
			pos2 = features[idx][2]        
			neg1 = features[idx][3]                    
			neg2 = features[idx][4]                                
			return user, pos1, pos2, neg1, neg2




