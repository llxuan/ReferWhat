import os
import os.path as osp
import torch
from tools import Config, setup_seed, initializeWeights
from BatchData import VD_REGLoader, get_iterator, get_data_file, load_vocab
from Trainer import REGTrainer
from REGer import REGModel

# choose GPU
torch.set_num_threads(3)
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
setup_seed(1234)

# read config
cfg = Config('./configs/BaseModel.yml')

# set datapath
data_root = cfg.DATA_PATH
data_cfg = cfg.DATA
data_path = osp.join(data_root, data_cfg.DATA_SET, data_cfg.SPLIT + '_split')

# load vocab
vocab = load_vocab(data_path)
print('vocab_length:', len(vocab))

# load dataset
train_data = get_iterator(VD_REGLoader(vocab, cfg, split ='train'), cfg.TRAIN.T_BS)
eval_data = get_iterator(VD_REGLoader(vocab, cfg, split ='val'), cfg.TRAIN.V_BS)
testA_data = get_iterator(VD_REGLoader(vocab, cfg, split ='testA'), cfg.TRAIN.V_BS)
testB_data = get_iterator(VD_REGLoader(vocab, cfg, split ='testB'), cfg.TRAIN.V_BS)

# init model
model_cfg = cfg.MODEL
model = REGModel(vocab, cfg).cuda()

# training
trainer = REGTrainer(model, cfg)
trainer.train_and_eval(train_data, eval_data)



