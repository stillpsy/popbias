# dataset name 
'''
dataset = 'ml-1m'
assert dataset in ['ml-1m', 'pinterest-20', 'synthetic']
'''

# model name 
model = 'MLP'
assert model in ['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre']

# paths
main_path = './data'

'''
train_rating = main_path + '{}.train.rating'.format(dataset)
test_rating = main_path + '{}.test.rating'.format(dataset)
test_negative = main_path + '{}.test.negative'.format(dataset)
'''

model_path = './models/'
MLP_model_path = model_path + 'MLP.pth'
'''
GMF_model_path = model_path + 'GMF.pth'
NeuMF_model_path = model_path + 'NeuMF.pth'
'''

