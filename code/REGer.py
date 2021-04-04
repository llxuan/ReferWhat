import torch
from torch import nn
from models import VExtractor,HRNN, MaxMarginLoss, MLP2, Attn ,maskedCEL
from evaluation import Simulator, Cider
import random

class REGModel(nn.Module):
    def __init__(self, vocab, cfg):
        super(REGModel, self).__init__()
        self.cfg = cfg
        self.rmax = cfg.R_MAX
        self.smax = cfg.S_MAX
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.bod = self.word2idx[self.cfg.BOD]
        self.bos = self.word2idx[self.cfg.BOS]
        self.eos = self.word2idx[self.cfg.EOS]
        self.pad = self.word2idx[self.cfg.PAD]
        self.noc = self.word2idx[self.cfg.NOC]
        self.loc = self.word2idx[self.cfg.LOC]
        self.visextractor = VExtractor(cfg)
        self.v_drop_layer = nn.Dropout(p=self.cfg.MODEL.CNN.VISUAL_DROPOUT)

        self.hrnn = HRNN(vocab, cfg)
        self.bi_max = int(self.rmax/2)
        self.batch_size = 1
        self.reg_mask_ref = torch.arange(10).view(1, -1).repeat(cfg.TRAIN.T_BS, 1).cuda()   # 10 is the default att_num
        self.attn = Attn('concat', cfg.MODEL.CNN.EMBD_DIM)
        self.v_embd = nn.Linear(2*cfg.MODEL.CNN.EMBD_DIM, cfg.MODEL.CNN.EMBD_DIM)

        if cfg.RL_Train:
            
            self.simulator = Simulator()
            # batch_size
            self.round_mask_ref = torch.LongTensor(sum([[a] * (self.smax -1) \
                                                    for a in range(self.bi_max)],
                                                     [])).view(1, -1).repeat(cfg.TRAIN.T_BS, 1).cuda()  
        else:
            self.mmi_loss = MaxMarginLoss(cfg.TRAIN.SL.Margin, cfg.TRAIN.SL.Lama, cfg.TRAIN.SL.Lamb)
       

    def forward(self, data):
        vis_feats = data['vis_feats'].cuda()
        regs_num = data['regs_num'].cuda()
        self.batch_size = vis_feats.size(0)
        vis_emb = self.visextractor(vis_feats)
        if self.cfg.RL_Train:
            '''
            Explaination of 
            x: 'a b$c d'
            x.split('$'): ['a b','c d']
            a.split(' ') for a in x.split('$') [['a','b'],['c','d']]
            List [[['w_1','w_2'],['w_3','w_4']], [['w_1','w_2'],['w_3','w_4']]] batch_size: RE_num, e_num
            '''

            entities = [[a.split(' ') for a in x.split('$')] for x in data['entities']]   
            REs = [x.split('$') for x in data['REs']]
            result = self.gen_append_rl_loop(vis_emb, regs_num, entities, REs) 
        else: 
            dial_token = data['dial_token'].cuda()
            dial_len = data['dial_len'].cuda()
            
            neg_vis_feat = data['neg_vis_feats'].cuda()
            neg_dial_token = data['neg_dial_token'].cuda()
            neg_dial_len = data['neg_dial_len'].cuda()
            neg_regs_num = data['neg_regs_num'].cuda()
            neg_vis_emb = self.visextractor(neg_vis_feat)
            pos_pos_pred = self.gen_append_sl_loop(vis_emb,regs_num, dial_token, dial_len)
            neg_pos_pred = self.gen_append_sl_loop(neg_vis_emb, neg_regs_num, dial_token, dial_len)
            pos_neg_pred = self.gen_append_sl_loop(vis_emb, regs_num, neg_dial_token, neg_dial_len)
            pred_target = torch.LongTensor(self.batch_size, pos_neg_pred.size(1), self.smax).zero_().cuda()
            for i in range(pred_target.size(1)):
                pred_target[:, i] = dial_token[:, 2*i]
            result = self.mmi_loss(pos_pos_pred, neg_pos_pred, pos_neg_pred, pred_target)

        return result

    # functions for append model
    def _init_bod(self):
        bod = torch.LongTensor([self.bod]).unsqueeze(0).cuda()
        noc = torch.LongTensor([self.noc]).unsqueeze(0).cuda()

        fact_label = torch.cat([bod, noc], 1).unsqueeze(2).repeat(self.batch_size, 1, 1)
        fact_len = torch.LongTensor(self.batch_size,2).fill_(1).cuda()
        
        return fact_label, fact_len


    def gen_append_per(self, visfeats, regs_num, fact, fact_len, his_h = None, fact_h = None, target = None, target_len = None, r = 0):
        REGer_emb, fact_h = self.hrnn.enc_reger_perround(fact[:,0,:], fact_len[:,0], fact_h)
        REUer_emb = self.hrnn.enc_reuer_perround(fact[:, 1, :], fact_len[:, 1])
        his_emb , his_h = self.hrnn.enc_his(torch.cat([REGer_emb, REUer_emb], 1), his_h)
       
        (v_r, v_diff) = visfeats
        attn_num = v_diff.size(1)
        self.batch_size = v_diff.size(0)
        regs_mask_ref = self.reg_mask_ref[:self.batch_size]
        regs_mask = (regs_mask_ref < regs_num.unsqueeze(1)).view(self.batch_size, -1)

        if r == 0:
            w = 1 / attn_num
            v_att_w = torch.FloatTensor(attn_num, self.batch_size).fill_(w).cuda()
        else:
            v_att_w = self.attn.negforward(his_emb.transpose(0, 1), v_diff.transpose(0, 1)).transpose(0, 1)

        v_att_w.data.masked_fill_(~regs_mask.transpose(0,1).data, 0)
        # v_att_norm_w batch x 1 x attn_nums
        v_att_norm_w = torch.div(v_att_w, torch.clamp(torch.sum(v_att_w, dim = 0), min = 0.0000001)).transpose(0,1).unsqueeze(1)
        # batch x 1 x attn_num bmm batch x attn_num x 1024 -> batch x 1 x 1024
        v_diff_attn = torch.bmm(v_att_norm_w, v_diff).squeeze(1)
        visfeats = self.v_embd(torch.cat([v_r, v_diff_attn], -1))
            
        visfeats = self.v_drop_layer(visfeats)
        pred = self.hrnn.decoder(visfeats, his_emb, target, target_len)

        return pred, his_h, fact_h

    def gen_append_sl_loop(self, vis_feats, regs_num, dial, dial_len):
        all_rounds_pred = []
        his_h = None
        fact_h = None

        for i in range(self.bi_max):
            if i == 0:
                fact_label, fact_len = self._init_bod()
            else:
                fact_label = dial[:, 2 * (i - 1):2*i, :]
                fact_len = dial_len[:, 2 * (i - 1):2*i]
            pred, his_h, fact_h = self.gen_append_per(vis_feats, regs_num, fact_label, fact_len, 
                                                his_h, fact_h, dial[:, 2*i, :], dial_len[:, 2*i], r= i)
            all_rounds_pred.append(pred)
            
        gen_pred = torch.stack(all_rounds_pred, 0).permute(1,0,2,3)
        return gen_pred

    # functions for rl append model
    def get_self_critical_reward(self, data_gts, gen_result):

        batch_size = len(gen_result)  # batch_size = sample_size * seq_per_img
        res = {}
        for i in range(batch_size):
            res[i] = [' '.join(gen_result[i])]
        batch_reward = 0
        gts = {}
        for i in range(batch_size):
            gts[i] = data_gts[i]

        if self.cfg.TRAIN.RL.Cider_Reward_Weight > 0:
            cider, cider_scores = Cider().compute_score(gts, res)
            cider_scores = self.cfg.TRAIN.RL.Cider_Reward_Weight * torch.FloatTensor(cider_scores)
            batch_reward = cider
        else:
            cider_scores = 0

        rewards = cider_scores.unsqueeze(1).repeat(1, self.bi_max)

        return batch_reward, rewards
    
    def immediate_reward(self, reward, batch_locate):
        i_reward = torch.FloatTensor(reward.size()).zero_()
        for i in range(self.batch_size):
            locate = batch_locate[i]
            for j in range(self.bi_max):
                temp = reward[i, j]
                if locate:
                    i_reward[i, j] = temp
                else:
                    i_reward[i, j] = min(-1 + temp, -0.1)
        return i_reward

    def remove_duplicate(self, pred_tensor):
        '''
        removed duplicate sentence, and static the duplicate word in other round
        :param pred_tensor:
        :return:
        '''
        removed_tensor = torch.LongTensor(pred_tensor.size()).zero_().cuda()
        duplicate_tensor = torch.FloatTensor(pred_tensor.size(0), pred_tensor.size(1)).zero_().cuda()
        for i, pred in enumerate(pred_tensor):
            _dup = []
            for j, round in enumerate(pred):
                if j == 0:
                    removed_tensor[i,j] = round
                    temp = list(round)
                else:
                    m = 0
                    # remove duplicate sent in train
                    for k, word in enumerate(round):
                        if word in temp:
                            if word == self.pad or word == self.eos:
                                pass
                            else:
                                duplicate_tensor[i, j] = 1.0
                        else:
                            temp.append(word)
                            m += 1
                    if m > 0:
                        removed_tensor[i, j] = round
        return removed_tensor, duplicate_tensor
    
    def gen_append_rl_loop(self, vis_feats, regs_num, entities, REs):
        #print('REs',REs)
        
        self.batch_size = vis_feats[0].size(0)
        
        all_preds = [[] for _ in range(self.batch_size)]
        all_locates = [False] * self.batch_size
        all_pred_probs = []
        all_pred_tokens = []
        all_dec_out = []
        round_mask_ref = self.round_mask_ref[:self.batch_size]
        pred_round_num = torch.LongTensor(self.batch_size).fill_(self.bi_max).cuda()
        score = 0
        for i in range(self.bi_max):
            if i == 0:
                fact_label, fact_len = self._init_bod()
                his_h = None
                fact_h = None
            # pred 'a man'
            # pred_token '1 2 EOS'
            # pred_len 3
            # pred_prob 3xlen_voc
            # dec_out output of decoder GRU
            (pred, pred_token, pred_len, pred_prob, dec_out), his_h, fact_h = self.gen_append_per(
                vis_feats, regs_num, fact_label, fact_len, his_h, fact_h, r = i)
            all_pred_probs.append(pred_prob)
            all_pred_tokens.append(pred_token)
            all_dec_out.append(dec_out)
            bos_tensor = torch.LongTensor(self.batch_size, 1).fill_(self.bos).cuda()
            fact_label = torch.cat([bos_tensor, pred_token], 1)
            fact_len = (pred_len + 1).cuda()  
            len_size = fact_label.size(1)
            noc_tensor = torch.LongTensor(self.batch_size, 1, len_size).zero_().cuda()
            fact_label = torch.cat([fact_label.unsqueeze(1), noc_tensor],1)
            
            fact_len = torch.cat([fact_len.unsqueeze(1), torch.LongTensor(self.batch_size, 1).fill_(1).cuda()], 1)
            
            for j in range(self.batch_size):
                fact_label[j,1,0] = self.noc
                if not all_locates[j]:
                    all_preds[j].append(pred.copy()[j])
                    respond, match_num = self.simulator.respond(entities[j], all_preds[j])
                    if respond == 'locate the object':
                        all_locates[j] = True
                        pred_round_num[j] = i + 1
                        score += 1  
                        fact_label[j,1,0] = self.loc

        all_pred_tokens_tensor = torch.stack(all_pred_tokens, 0).transpose(1,0)
        all_pred_probs_tensor = torch.stack(all_pred_probs, 0).transpose(1,0)
        all_dec_out_tensor = torch.stack(all_dec_out, 0).transpose(1,0)
        cum_rewards = torch.FloatTensor(self.batch_size, self.bi_max, self.smax-1).zero_().cuda() 
        success_mask = torch.FloatTensor(self.batch_size, self.bi_max, self.smax-1).zero_().cuda()  
        round_mask = (round_mask_ref < pred_round_num.unsqueeze(1)).view(self.batch_size, self.bi_max, -1)
        all_pred_tokens_tensor.data.masked_fill_(~round_mask.data, 0)
        pad_mask = torch.ge(all_pred_tokens_tensor, 1)
 
        if self.cfg.TRAIN.RL.Cider_Reward_Weight > 0:
            batch_cider, cider = self.get_self_critical_reward(REs, all_preds)
            i_reward = self.immediate_reward(cider, all_locates)
            cum_rewards = i_reward.unsqueeze(2).repeat(1, 1, self.smax-1).cuda()  
            for i in range(self.batch_size):
                if all_locates[i]:
                    success_mask[i][:][:] = 1
                else:
                    success_mask[i][:][:] = 0  

        cum_rewards.data.masked_fill_(~pad_mask.data, 0)
        success_mask.data.masked_fill_(~pad_mask.data, 0)

        score = 1.0 * score / self.batch_size

        return (score, batch_cider, all_dec_out_tensor, all_pred_probs_tensor, pad_mask, cum_rewards, success_mask)
