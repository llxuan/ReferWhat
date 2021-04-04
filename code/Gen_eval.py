from __future__ import division
import logging
import os
import os.path as osp
import json
import nltk
import time
import torch
from torch.autograd import Variable
from tools import Config, setup_logging
from evaluation import Simulator, REGEvaluation
from BatchData import VD_REGLoader, get_iterator, get_data_file,load_vocab
from REGer import REGModel


os.environ["CUDA_VISIBLE_DEVICES"]="3"
MODE = 'append'
DATA_SET = 'refcoco'
DATA_SPLIT = 'unc_1222'
DATA_TYPE = 'combine'
FEATS_FILE = 'resnet152'
SAVE_DIR = 'Adam_0.0001_test'
SAVE_PATH = os.path.join('..', MODE + '_model_save', DATA_SET, DATA_SPLIT,
                         DATA_TYPE, FEATS_FILE, SAVE_DIR)
#CHECKPOINT = 'max_cider_ckp'
#CHECKPOINT = '115_ckp'
CHECKPOINT = 'min_loss_ckp'
ab = 'A'
SPLIT = 'test' + ab
BEAM_SIZE = 2
LOG_FILE = 'log_gen_' + SPLIT + '_' + str(BEAM_SIZE) +'.txt'
JSON_FILE = 'pred_' + SPLIT + '_' + str(BEAM_SIZE) + '.json'
EVAL_FILE = 'eval_' + SPLIT + '_' + str(BEAM_SIZE) + '.json'

class Predictor(object):
    def __init__(self, model, data, data_cfg, save_path):
        self.model = model
        self.test_loader = data
        self.save_path = save_path
        self.data_cfg = data_cfg
        if self.model.cfg.MODEL.ENCREU:
            self.noc = torch.LongTensor([[self.model.noc]]).cuda() 

    def predict(self, item, beam_size = 3):

        data = self.test_loader.__getitem__(item)

        vis_feats = data['vis_feats'].unsqueeze(0).cuda()
        entities = data['entities']
        REs = data['REs']
        ref_dial = data['ref_dial']
        ref_id = data['ref_id']
        dial_id = data['dial_id']
        regs_num = data['regs_num']
        #print(regs_num)
        simulator = Simulator()
        speaker_list = []
        dialog_list = []
        atten_weight = []
        vis_feats = self.model.visextractor(vis_feats)
        regs_mask_ref = torch.arange(10).view(1, -1).repeat(1, 1).cuda()   # 10 is the default att_num
        regs_mask = (regs_mask_ref < regs_num).view(1, -1)

        if self.model.cfg.MODEL.VIS_METHOD == 'atten':
            (v_r, v_diff) = vis_feats

        for i in range(self.model.bi_max):
            if i == 0:
                fact_label, fact_len = self.model._init_bod()
                fact_label = torch.LongTensor([[self.model.bod]]).cuda()
                his_h = None
                fact_h = None
            if self.model.cfg.MODEL.ENCREU:
                REGer_emb, fact_h = self.model.hrnn.enc_reger_perround(fact_label, fact_len[:, 0], fact_h)
                REUer_emb = self.model.hrnn.enc_reuer_perround(self.noc, fact_len[:, 1])
                his_emb , his_h = self.model.hrnn.enc_his(torch.cat([REGer_emb, REUer_emb], 1), his_h)
            else:
                fact_emb, fact_h = self.model.hrnn.enc_reger_perround(fact_label, fact_len, fact_h)
                #print(fact_emb)
                if self.model.cfg.MODEL.HRED:
                    his_emb, his_h = self.model.hrnn.enc_his(fact_emb, his_h)
                else:
                    his_emb = fact_emb.unsqueeze(1)
            if self.model.cfg.MODEL.VIS_METHOD == 'atten':
                attn_num = v_diff.size(1)
                self.batch_size = v_diff.size(0)
                #his_emb = torch.relu(self.model.fc2(torch.relu(self.model.fc1(his_emb))))
                if i == 0:
                    w = 1/attn_num
                    #print(attn_num)
                    v_att_w = torch.FloatTensor(attn_num, self.batch_size).fill_(w).cuda()
                else:
                    v_att_w = self.model.attn.negforward(his_emb.transpose(0,1), v_diff.transpose(0,1)).transpose(0,1)
                
                v_att_w.data.masked_fill_(~regs_mask.transpose(0,1).data, 0)
                # v_att_norm_w batch x 1 x attn_num
                v_att_norm_w = torch.div(v_att_w ,torch.clamp(torch.sum(v_att_w, dim = 0), min = 0.0000001)).transpose(0,1).unsqueeze(1)
                atten_weight.append(v_att_norm_w.data.cpu().numpy().tolist())
                # batch x 1 x attn_num bmm batch x attn_num x 1024 -> batch x 1 x 1024
                v_diff_attn = torch.bmm(v_att_norm_w, v_diff).squeeze(1)
                vis_feats = self.model.v_embd(torch.cat([v_r, v_diff_attn], -1))
            
            pred, token = self.model.hrnn.decoder(vis_feats, his_emb, training = False, beam_size = beam_size)
            speaker_list.append(pred)
            listener, _ = simulator.respond(entities, speaker_list)
            dialog_list.append(pred)
            dialog_list.append(listener)
            if listener == 'locate the object':
                break
            s_len = len(token) + 1
            if self.model.cfg.MODEL.ENCREU:
                fact_len = torch.LongTensor([[s_len, 1]]).cuda()
                fact_label = torch.LongTensor(s_len).cuda()
                fact_label[0] = self.model.bos
                for j in range(1, s_len):
                    fact_label[j] = token[j - 1]
                fact_label = fact_label.unsqueeze(0)
            else:
                fact_len = torch.LongTensor([s_len]).cuda()
                fact_label = torch.LongTensor(s_len).cuda()
                fact_label[0] = self.model.bos
                for j in range(1, s_len):
                    fact_label[j] = token[j - 1]
                fact_label = fact_label.unsqueeze(0)

        # print( 'ref id %s:' % (ref_id))
        # print('Pred: %s' % (speaker_list))
        # print('Dialog Pred: %s' %(dialog_list))
        # print('Dialog: %s' % (dialog))
        # print('RE: %s' % (sent))
        # print('Entity: %s' % (entity))
        # print('\n')
        return speaker_list, dialog_list, REs, entities, ref_dial, ref_id, dial_id, atten_weight

    def predict_all(self,beam_size = 3):
        setup_logging(os.path.join(self.save_path, 'log', LOG_FILE))
        all_refer = {}

        l = len(self.test_loader)

        for i in range(l):
            pred_sent, dialog_list, sents, entities, dialog, ref_id, dial_id, atten_weight= self.predict(i, beam_size)
            all_refer[ref_id] = {
                'speaker': pred_sent,
                'dialog': dialog_list,
                'REs': sents,
                'attn_w': atten_weight
            }
            logging.info('%d/%d, ref id %s:' % (i+1, l, ref_id))
            logging.info('dial_id %s:' % (ref_id))
            logging.info('Pred: %s' % (pred_sent))
            logging.info('Dialog Pred: %s' % (dialog_list))
            logging.info('Dialog: %s' % (dialog))
            logging.info('REs: %s' % (sents))
            logging.info('Entities: %s' % (entities))

        return all_refer


eval_path = os.path.join(SAVE_PATH, 'eval')
os.makedirs(eval_path,  exist_ok=True)

#'''
config_path = osp.join(SAVE_PATH, 'config', 'trainer_config.yml')
cfg = Config(config_path)
print('load model from ', SAVE_PATH)
check_point = torch.load(os.path.join(SAVE_PATH, 'checkpoints', CHECKPOINT))#,map_location = lambda _1,_2,:_1)

# load dataset
data_root = cfg.DATA_PATH
data_cfg = cfg.DATA
data_path = osp.join(data_root, data_cfg.DATA_SET, data_cfg.SPLIT + '_split')
vocab = load_vocab(data_path)
print('vocab_length:', len(vocab))
test_loader = VD_REGLoader(vocab, cfg, split = SPLIT)

model_cfg = cfg.MODEL
model = REGModel(vocab, cfg).cuda()
model.load_state_dict(check_point['state_dict'])
model.eval()
print('model load finish')

predor = Predictor(model, test_loader, data_cfg, SAVE_PATH)


all_refers = predor.predict_all(BEAM_SIZE)

json.dump(all_refers, open(os.path.join(eval_path, JSON_FILE), 'w'))
#'''
# all_refers =json.load(open(os.path.join(eval_path, JSON_FILE),'r'))
#eval predictions

refer = {}
pred = {}
dialog = {}
for key, item in all_refers.items():
    refer[key] = item['REs']
    pred[key] = item['speaker']
    dialog[key] = item['dialog']

vd_re_eval = REGEvaluation(refer, pred, dialog)
shuffle_eval_result = vd_re_eval.eval_shuffle()
print('shuffle overall metric:', shuffle_eval_result)
vd_re_eval.sucess_rate()
sucess = vd_re_eval.sucess
print('dialog sucess rate:', sucess)
print('dialog fail rate:', 1-sucess)
print('dialog sucess distribute:', vd_re_eval.round_count)

refToEval = vd_re_eval.refToEval
for ref_id, speaker in pred.items():
    refToEval[str(ref_id)]['speaker'] = speaker
    refToEval[str(ref_id)]['REs'] = refer[ref_id]

with open(os.path.join(eval_path, EVAL_FILE), 'w') as outfile:
    json.dump({'shuffle_eval_result': shuffle_eval_result,  'refToEval': refToEval}, outfile)



