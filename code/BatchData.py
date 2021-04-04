#prepare for data that need for training
import os
import os.path as osp
import nltk
import h5py
import json
import random
import torch
import torch.utils.data as data
from torch.distributions.categorical import Categorical

answer_dict = {'no': torch.Tensor([0]),
               'yes': torch.Tensor([1])}
               
def get_location_feats(box, W, H):
    '''
    get five dimension location feats
    :param box: bbox  [x1, y1, w, h]
    :param W:   image width
    :param H:   image height
    :return:    x1/W , y1/H, x2/W, y2/H, w*h/W*H
    '''
    return [box[0] / W, box[1] / H, max(box[0] + 1, box[0] + box[2] - 1) / W,
            max(box[1] + 1, box[1] + box[3] - 1) / H, box[2] * box[3] / (W * H)]

def load_vocab(data_path):
    vocab_file = osp.join(data_path, 'c_vocab.json')
    vocab_dict = json.load(open(vocab_file, 'r'))
    vocab_list = vocab_dict['vocab']
    return vocab_list

def get_data_file(data_root, split):
    '''
    :param data_root:
    :param split:
    :return:
    '''
    data_file = osp.join(data_root, 'dump_c_' + split + '.json')
    return data_file

class VD_REGLoader(data.Dataset):
    def __init__(self, vocab, cfg, split ='train'):
        '''
        :param vocab:
        :param data_root:
        :param cfg:
        :param split:
        '''
        self.cfg = cfg
        self.datacfg = cfg.DATA
        self.s_max = cfg.S_MAX
        self.r_max = cfg.R_MAX
        self.bi_max = self.r_max // 2
        self.split = split
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.bod = self.word2idx[self.cfg.BOD]
        self.bos = self.word2idx[self.cfg.BOS]
        self.eos = self.word2idx[self.cfg.EOS]
        self.unk = self.word2idx[self.cfg.UNK]
        self.pad = self.word2idx[self.cfg.PAD]
        self.loc = self.word2idx[self.cfg.LOC]
        self.noc = self.word2idx[self.cfg.NOC]

        data_root = cfg.DATA_PATH
        data_path = osp.join(data_root, self.datacfg.DATA_SET, self.datacfg.SPLIT + '_split')
        data_file = get_data_file(data_path, self.split)
        data = json.load(open(data_file, 'r'))
        self.imgs_dict = data['imgs']
        self.anns_dict = data['anns']
        self.refs_dict = data['refs']
        self.dials_dict = data['dials']
        self.sents = data['sents']
        self.entities = data['entities']
        self.img2refs_dict = data['imgs2refs']
        self.img2cands_dict = data['imgs2cands']
        self.ref_ids_list = data['ref_ids']
        self.dial_ids_list = data['dial_ids']

        self.dialogs_use_tag_dict = {}
        for dial_id, dialog in self.dials_dict.items():
            self.dialogs_use_tag_dict[dial_id] = 0

        self.feat_root = osp.join(data_root, self.datacfg.DATA_SET, 'feats')
        feat_idx_f = osp.join(self.feat_root, 'id_to_h5_' + self.datacfg.FEATS_FILE +'.json')
        id_to_ix = json.load(open(feat_idx_f,'r'))
        self.img_idx = id_to_ix['image_h5_id']
        self.reg_idx = id_to_ix['region_h5_id']
        self.featfile = osp.join(self.feat_root, 'feats_' + self.datacfg.FEATS_FILE +'.h5')

    def __getitem__(self, item):
  

        ref_id = self.ref_ids_list[item]
        dial_ids = self.refs_dict[str(ref_id)]['dial_ids']
        max_fetch = max([self.dialogs_use_tag_dict[str(i)] for i in dial_ids])
        sample_dialog_ids = [i for i in dial_ids if self.dialogs_use_tag_dict[str(i)] < max_fetch]
        if sample_dialog_ids == []:
            sample_dialog_ids = dial_ids
        dial_id = random.choice(sample_dialog_ids)
        self.dialogs_use_tag_dict[str(dial_id)] += 1

        ref_info = self.refs_dict[str(ref_id)]
        sent_ids = ref_info['re_ids']
        entities = []
        for sent_id in sent_ids:
            entities.append(self.entities[str(sent_id)])
        ref_feat, ref_regs_num = self.fetch_feats(ref_info, ref_id, ref_id)

        ref_dial = self.dials_dict[str(dial_id)]['dial']
        ref_dial_token = self.dials_dict[str(dial_id)]['dial_token']
        ref_dial_round = self.dials_dict[str(dial_id)]['rounds']
        data = {}
        data['vis_feats'] = ref_feat
        data['regs_num'] = ref_regs_num
        if self.split in ['train', 'val']:
            ref_dial_token, ref_dial_len = self.create_target(ref_dial_token, ref_dial_round)      
            data['dial_token'] = ref_dial_token
            data['dial_len'] = ref_dial_len
            # for reinforcement learning     
            if self.cfg.RL_Train: 
                data['entities'] = '$'.join([' '.join(x) for x in entities])
                data['REs'] = '$'.join(self.sents[str(x)] for x in sent_ids)    
            else:    
                # negtive sampling  
                neg_ann_id, neg_dial_id = self.sample_neg_id(ref_info, ref_id)  
                neg_vis_feats, neg_regs_num = self.fetch_feats(ref_info, str(neg_ann_id), ref_id)
                neg_dial = self.dials_dict[str(neg_dial_id)]['dial']
                neg_dial_token = self.dials_dict[str(neg_dial_id)]['dial_token']
                neg_dial_round = self.dials_dict[str(neg_dial_id)]['rounds']
                neg_dial_token, neg_dial_len = self.create_target(neg_dial_token, neg_dial_round)
                data['neg_vis_feats'] = neg_vis_feats
                data['neg_dial_token'] = neg_dial_token
                data['neg_dial_len'] = neg_dial_len
                data['neg_regs_num'] = neg_regs_num
        elif self.split in ['testA', 'testB']:
            sents = []
            for id in sent_ids:
                sents.append(self.sents[str(id)])
            data['REs'] = sents
            data['entities'] = entities
            data['ref_dial'] = ref_dial
            data['ref_id'] = ref_id
            data['dial_id'] = dial_id

        return data

    def __len__(self):

        return len(self.refs_dict)

    def fetch_feats(self, ref_info, ann_id, ref_id):
        ann_bbox = self.anns_dict[str(ann_id)]['bbox']
        img_id = ref_info['img_id']
        img_info = self.imgs_dict[str(img_id)]
        img_W = img_info['w']
        img_H = img_info['h']
        ref_loc_feat = torch.Tensor(get_location_feats(ann_bbox, img_W, img_H))

        feats_f = h5py.File(self.featfile, 'r')
        img_feats = feats_f['image_feats']  # np.array image feats image_h5_id to image_feats
        region_feats = feats_f['region_feats']  # np.array region feats region_h5_id to region feats
        ref_feat = torch.Tensor(region_feats[self.reg_idx[str(ann_id)]])
        img_feat = torch.Tensor(img_feats[self.img_idx[str(img_id)]])
        feats, regs_num = self.fetch_attn_feats(ref_info, img_feat, ref_feat, ref_loc_feat, ann_id, ref_id)
        
        return feats, regs_num

    def fetch_neighbour_ids(self, ref_info, ann_id, ref_id):
        img_id = ref_info['img_id']
        img_info = self.imgs_dict[str(img_id)]
        ref_cate = self.anns_dict[str(ann_id)]['category_id']
        ann_ids = img_info['cand_ids'].copy()       # int
        ann_ids.remove(int(ref_id))
        ref_ids = self.img2refs_dict[str(img_id)].copy()
        ref_ids.remove(int(ref_id))
        x, y, w, h = self.anns_dict[str(ann_id)]['bbox']
        rcx, rcy = x + w / 2, y + h / 2

        def fetch_key(ann_id):
            [x, y, w, h] = self.anns_dict[str(ann_id)]['bbox']
            ax, ay = x + w / 2, y + h / 2
            return (rcx - ax) ** 2 + (rcy - ay) ** 2

        ann_ids.sort(key=fetch_key)

        st_ann_ids,  dt_ann_ids,  = [],  []
        for ann_id in ann_ids:
            if self.anns_dict[str(ann_id)]['category_id'] == ref_cate:
                st_ann_ids.append(ann_id)
            else:
                dt_ann_ids.append(ann_id)

        st_ref_ids, dt_ref_ids, = [], []
        for ref_id in ref_ids:
            if self.anns_dict[str(ref_id)]['category_id'] == ref_cate:
                st_ref_ids.append(ref_id)
            else:
                dt_ref_ids.append(ref_id)

        return st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids
    
    

    def fetch_attn_feats(self, ref_info, img_feat, ref_feat, ref_loc_feat, ann_id, ref_id, attn_num = 10):
        ann_bbox = self.anns_dict[str(ann_id)]['bbox']
        feats_f = h5py.File(self.featfile, 'r')
        region_feats = feats_f['region_feats']
        visual_dim = ref_feat.size()[0]
        diff_ann_feat = torch.FloatTensor(visual_dim).zero_()
        diff_loc_feat = torch.FloatTensor(attn_num * 5).zero_()
        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = self.fetch_neighbour_ids(ref_info, ann_id, ref_id)
        [rcx, rcy, rw, rh] = ann_bbox

        st_num = min(attn_num, len(st_ann_ids))
        dt_num = 0
        loc_dim = 5
        att_feats = torch.FloatTensor(attn_num, 3*visual_dim + 2*loc_dim).zero_()
        for i in range(st_num):
            cand_id = st_ann_ids[i]
            cand_bbox = self.anns_dict[str(cand_id)]['bbox']
            [cx1, cy1, cw, ch] = cand_bbox
            cand_loc_diff=torch.FloatTensor([(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw,
                                                    (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
            cand_feat = torch.Tensor(region_feats[self.reg_idx[str(cand_id)]])
            cand_visual_feats = torch.cat([img_feat, ref_feat, ref_loc_feat, cand_feat - ref_feat, cand_loc_diff], 0)
            att_feats[i] = cand_visual_feats
        
        dt_num = min(attn_num - st_num, len(dt_ann_ids))
        for j in range(dt_num):
            cand_id = dt_ann_ids[j]
            cand_bbox = self.anns_dict[str(cand_id)]['bbox']
            [cx1, cy1, cw, ch] = cand_bbox
            cand_loc_diff=torch.FloatTensor([(cx1 - rcx) / rw, (cy1 - rcy) / rh, (cx1 + cw - rcx) / rw,
                                                    (cy1 + ch - rcy) / rh, cw * ch / (rw * rh)])
            cand_feat = torch.Tensor(region_feats[self.reg_idx[str(cand_id)]])
            cand_visual_feats = torch.cat([img_feat, ref_feat, ref_loc_feat,  cand_feat - ref_feat, cand_loc_diff], 0)
            att_feats[st_num + j] = cand_visual_feats
       
        regs_num = max(st_num + dt_num, 1)
       
        return att_feats, regs_num

    # borrow from https://github.com/mikittt/easy-to-understand-REG
    def sample_neg_id(self, ref_info, pos_ann_id):
        st_ref_ids, st_ann_ids, dt_ref_ids, dt_ann_ids = self.fetch_neighbour_ids(ref_info, pos_ann_id, pos_ann_id)
        # ann
        if len(st_ann_ids) > 0:     # and random.random() < 0.5:
            neg_ann_id = random.choice(st_ann_ids)
        elif len(dt_ann_ids) > 0:   # and random.random() < 0.5:
            neg_ann_id = random.choice(dt_ann_ids)
        # ref
        if len(st_ref_ids) > 0 :    #and random.random() < 0.5:
            neg_ref_id = random.choice(st_ref_ids)
        elif len(dt_ref_ids) > 0:   # and random.random() < 0.5:
            neg_ref_id = random.choice(dt_ref_ids)
        else:
            neg_ref_id = random.choice(self.ref_ids_list)
        cand_dialog_ids = self.refs_dict[str(neg_ref_id)]['dial_ids']
        neg_dialog_id = random.choice(cand_dialog_ids)

        return neg_ann_id, neg_dialog_id

    # sent_token list
    # BOS sent token EOS
    def sent_to_idx(self, sent_token, agent):
        sent = torch.LongTensor(self.s_max).zero_()
        sent_len = 0
        if agent == 'REGer':    # add BOS and EOS
            if sent_token is not []:
                sent[0] = self.bos
                for j, word in enumerate(sent_token):
                    sent[j + 1] = self.word2idx.get(word, self.unk)
                    if j == self.s_max - 3:
                        break
                sent[j + 2] = self.eos
                sent_len = j + 3
        elif agent == 'REUer':
            if sent_token is not []:
                if sent_token == ['can', 'not', 'locate', 'the', 'object']:
                    sent[0] = self.noc
                    sent_len = 1
                elif sent_token == ['locate', 'the', 'object']:
                    sent[0] = self.loc
                    sent_len = 1
                else:
                    for j, word in enumerate(sent_token):
                        sent[j] = self.word2idx.get(word, self.unk)
                        if j == self.s_max - 1:
                            break
                    sent_len = j + 1
        return sent, sent_len

    def sample_len(self, oringal_len):
        oringal_len = min(oringal_len, self.bi_max)
        categorical_dist = Categorical(probs=self.rate[oringal_len - 1])
        return categorical_dist.sample()
    
    # append situation verified
    def create_target(self, dialog_tokens, dial_round):
        '''
        :param word2idx:
        :param dialog:
        :param sent_max:
        :param round_max:
        :return:
        '''
        dialog = torch.LongTensor(self.r_max, self.s_max).zero_()
        dialog_len = torch.LongTensor(self.r_max).fill_(0)
       
        for i, sent_token in enumerate(dialog_tokens):
            if i % 2 == 0:
                agent = 'REGer'
            else:
                agent = 'REUer'
            dialog_i, dialog_i_len = self.sent_to_idx(sent_token, agent)
            #print(dialog_i, dialog_i_len)
            dialog[i] = dialog_i
            dialog_len[i] = dialog_i_len
            if i == self.r_max - 1:
                break
        return dialog, dialog_len


def get_iterator(data, batch_size=4, shuffle=True, num_workers=0, pin_memory=True):
    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory)
