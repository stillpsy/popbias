import torch
import torch.nn as nn
import torch.nn.functional as F 


class BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num, num_layers,
					dropout, model, GMF_model=None, MLP_model=None):
		super(BPR, self).__init__()
		self.dropout = dropout
		self.model = model
		self.MLP_model = MLP_model
        
		self.embed_user_MLP = nn.Embedding(user_num, factor_num)
		self.embed_item_MLP = nn.Embedding(item_num, factor_num)
		
		MLP_modules = []
		for i in range(num_layers):
			if i == 0:
				input_size = factor_num*2
				MLP_modules.append(nn.Dropout(p=self.dropout))
				MLP_modules.append(nn.Linear(input_size, input_size//2))
				MLP_modules.append(nn.ReLU())
			else:
				input_size = factor_num
				MLP_modules.append(nn.Dropout(p=self.dropout))
				MLP_modules.append(nn.Linear(input_size, input_size))
				MLP_modules.append(nn.ReLU())    
		self.MLP_layers = nn.Sequential(*MLP_modules)

		if self.model in ['MLP', 'GMF']:
			predict_size = factor_num 
		else:
			predict_size = factor_num * 2
		self.predict_layer = nn.Linear(predict_size, 1)

		self._init_weight_()

        
	def _init_weight_(self):
		""" We leave the weights initialization here. """
		if not self.model == 'NeuMF-pre':
			#nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			#nn.init.normal_(self.embed_item_GMF.weight, std=0.01)            
			nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight)
			nn.init.kaiming_uniform_(self.predict_layer.weight, 
									a=1, nonlinearity='sigmoid')

			for m in self.modules():
				if isinstance(m, nn.Linear) and m.bias is not None:
					m.bias.data.zero_()

        
        
	def forward_one_item(self, user, item):
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)            
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)
            
		if self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return prediction.view(-1)
    
	def forward(self, user, item_i, item_j):
		prediction_i = self.forward_one_item(user, item_i)
		prediction_j = self.forward_one_item(user, item_j)
        
		return prediction_i, prediction_j
    
    
    
    
    
    
class MF_BPR(nn.Module):
	def __init__(self, user_num, item_num, factor_num, num_layers,
					dropout, model, GMF_model=None, MLP_model=None):
		super(MF_BPR, self).__init__()
		self.dropout = dropout
		self.model = model
		self.MLP_model = MLP_model
        
		self.embed_user_MLP = nn.Embedding(user_num, factor_num)
		self.embed_item_MLP = nn.Embedding(item_num, factor_num)

        
		if self.model in ['MLP', 'GMF']:
			predict_size = factor_num 
		self._init_weight_()

        
	def _init_weight_(self):
		nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
		nn.init.normal_(self.embed_item_MLP.weight, std=0.01)
        
	def forward_one_item(self, user, item):
        
		embed_user_MLP = self.embed_user_MLP(user)
		embed_item_MLP = self.embed_item_MLP(item)            
                
		pred = torch.mul(embed_user_MLP, embed_item_MLP)
		pred = torch.sum(pred, 1)

		return pred.view(-1)        
        
    
	def forward(self, user, item_i, item_j):
		prediction_i = self.forward_one_item(user, item_i)
		prediction_j = self.forward_one_item(user, item_j)
        
		return prediction_i, prediction_j