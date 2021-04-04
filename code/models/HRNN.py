import random
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.nn.functional import log_softmax
from .model_utils import dynamicRNN
from .beam_search import Generator

class HRNN(nn.Module):
    def __init__(self, vocab, cfg):
        super(HRNN, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        
        self.rmax = cfg.R_MAX
        self.smax = cfg.S_MAX

        '''RNN config'''
        self.rnn_type = self.model_cfg.RNN.RNN_TYPE
        rnn_cell = nn.__dict__[self.rnn_type]
        self.hidden_size = self.model_cfg.RNN.RNN_HIDDEN
        self.rnn_drop_layer = nn.Dropout(p=self.model_cfg.RNN.RNN_DROPOUT)
        self.n_layers = self.model_cfg.RNN.RNN_LAYER

        self.feats_dim = self.model_cfg.CNN.EMBD_DIM

        decoder_in_size = self.model_cfg.WORD.EMBD_DIM + self.feats_dim
        self.speaker_decoder = rnn_cell(decoder_in_size, self.hidden_size,
        batch_first=True, num_layers = self.n_layers)
        
        '''vocab'''
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.word_classifier = nn.Linear(self.hidden_size, self.vocab_size)

        self.bod = self.word2idx[self.cfg.BOD]
        self.bos = self.word2idx[self.cfg.BOS]
        self.eos = self.word2idx[self.cfg.EOS]
        self.unk = self.word2idx[self.cfg.UNK]
        self.pad = self.word2idx[self.cfg.PAD]
        self.loc = self.word2idx[self.cfg.LOC]
        self.noc = self.word2idx[self.cfg.NOC]

        '''Embed config'''
        self.embed_size = self.model_cfg.WORD.EMBD_DIM
        self.w_drop_layer = nn.Dropout(p=self.model_cfg.WORD.EMBD_DROPOUT)
        self.embedder = nn.Embedding(self.vocab_size, self.embed_size)
   
        self.REGenc = rnn_cell(self.embed_size, self.hidden_size,  batch_first=True,num_layers = self.n_layers )
        self.REUenc = rnn_cell(self.embed_size, self.hidden_size, batch_first=True, num_layers=self.n_layers)
        self.h_enc = rnn_cell(2*self.hidden_size, self.hidden_size, batch_first=True, num_layers=self.n_layers)
        

    def enc_his(self, his, his_hidden):
        his = his.unsqueeze(1)
        his = torch.relu(his)
        his = self.rnn_drop_layer(his)
        if his_hidden is None:
            his_hidden = self.init_hidden(self.h_enc, self.batch_size)
        his_emb, history_h = self.h_enc(his, his_hidden)  # 1 x batch x hidden
        his_emb = torch.relu(his_emb)
        his_emb = self.rnn_drop_layer(his_emb)
        return his_emb, history_h
    
    def enc_reuer_perround(self, tokens, lens):
        reuer, hidden = self.hdynamicRNN(self.REUenc, tokens, lens, hidden = None)
        return reuer
    
    def enc_reger_perround(self, tokens, lens, hidden = None):
        reger, hidden = self.hdynamicRNN(self.REGenc, tokens, lens, hidden)
        return reger, hidden

    def init_hidden(self, rnn, batch_size):
        '''Initial  rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert batch_size != 0, 'Observe something to infer batch size.'
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (weight.new(rnn.num_layers, batch_size, rnn.hidden_size).zero_(),
                    weight.new(rnn.num_layers, batch_size, rnn.hidden_size).zero_())
        else:
            return weight.new(rnn.num_layers, batch_size, rnn.hidden_size).zero_()

    def hdynamicRNN(self, rnn, tokens, lens, hidden = None):
        '''
        '''
        # cannot model the relationship between 
        mask = (lens > 0)  
        self.batch_size = lens.size(0) 
        actual = tokens[mask]                                               # -> (? , 11)
        embs = torch.zeros(self.batch_size, self.embed_size).cuda()
        if actual.size(0) > 0:
            actual_lens = lens[mask]                                            # -> (?, )
            actual_embs = self.w_drop_layer(self.embedder(actual))              # -> (?, 11, 300)
            if hidden is None:
                hidden = self.init_hidden(rnn, actual.size(0)).cuda()
            actural_ = dynamicRNN(rnn, actual_embs, actual_lens, initialState= hidden)
            hidden = None  
            embs[mask] = actural_.squeeze()                                 # -> (b, 6, 512)
        return embs, hidden


    def decoder(self, vis_feat, his_emb, target = None, target_len = None, training = True, beam_size = 3):
        '''

        :param visual_feats: # batch x 1 x feats_dim
        :param speak_embed:  # batch x 1 x [max_sent_len - 1]x emd_dim , only the first sent
        :return:
        '''
        batch_size = vis_feat.size(0)
        
        # reinforcement training 
        if target is None and training:# and self.cfg.RL_Train:
            # begin generation
            batch_speaker = []          # id
            pred_len = torch.LongTensor(batch_size).fill_(0)  # len
            pred_prob = []     # prob
            rnn_output = []
            vis_feat = vis_feat.unsqueeze(1)
            word_emb_drop = his_emb
            rnn_input = torch.cat([vis_feat, word_emb_drop], 2)
            state = None

            for i in range(self.smax - 1):  # remove bos
                rnn_out, new_state = self.speaker_decoder(rnn_input, state)
                rnn_output.append(rnn_out.squeeze(1))
                output = self.word_classifier(rnn_out.squeeze(1))  # batch_size x vocab_size
                logits = log_softmax(output, dim=1)  # batch_size x vocab_size
                if random.random() < self.model_cfg.RL.SAMPLE_RATE:
                    categorical_dist = Categorical(logits=logits.detach())
                    word = categorical_dist.sample()
                    log_prob = logits.gather(1, word.unsqueeze(1))
                else:
                    log_prob, word_ = logits.topk(1, 1)
                    word = word_.squeeze()
                state = new_state
                word_emb = self.embedder(word.unsqueeze(1))
                word_emb_drop = self.w_drop_layer(word_emb)
                rnn_input = torch.cat([vis_feat, word_emb_drop], 2)
                batch_speaker.append(word)
                pred_prob.append(log_prob.squeeze())
            pred_token = torch.stack(batch_speaker, 0).permute(1, 0)
            #print(pred_token)
            # token         1 2 4 5 E 8 E
            # token== eos   0 0 0 0 1 0 0
            # cumsum(1)     0 0 0 0 1 1 2
            # cumsum(1)>=1  0 0 0 0 1 1 1
            # cat           0 0 0 0 0 1 1
            # token         1 2 4 5 E 0 0
            def tensor_rshift(a):
                lpad = torch.zeros((a.size(0), 1), dtype=a.dtype)
                lpad = lpad.cuda() if a.is_cuda else lpad
                return torch.cat((lpad, a[:, :-1]), dim = 1)
            pred_token[tensor_rshift((pred_token == self.eos).cumsum(1) >= 1)] = 0   # with eos
            # must be > =
            pred = []

            for i, batch_sent in enumerate(torch.stack(batch_speaker, 0).permute(1, 0)):
                word_pred = []
                for j, batch_word in enumerate(batch_sent):
                    if batch_word == self.eos:
                        break
                    if j == self.smax - 2 :
                        pred_token[i][j] = self.eos
                        break
                    if batch_word == self.pad:
                        continue
                    word_pred.append(self.vocab[batch_word])       # without eos

                sents = ' '.join(word_pred)
                pred_len[i] = len(word_pred) + 1          # with eos
                pred.append(sents)
            pred_prob = torch.stack(pred_prob, 0).transpose(1, 0)  # batch_size x self.smax x vocab_size without bos
            dec_out = torch.stack(rnn_output, 0).permute(1, 0, 2)  #  batch_size x self.smax x hidden
            return (pred, pred_token, pred_len, pred_prob, dec_out)#, history_h
        # supervised training
        elif training:
            mask = (target_len > 0)  
            actual = target[mask]                                               # -> (? , 11)
            word_pred = torch.zeros(self.batch_size, self.smax, self.vocab_size).cuda()
            if actual.size(0) > 0: 
                actual_lens = target_len[mask]                                            # -> (?, )
                actual_embs = self.w_drop_layer(self.embedder(actual))              # -> (?, 11, 300)
                actural_vis_feat = vis_feat[mask]
                guid = actural_vis_feat.unsqueeze(1).repeat(1, self.smax, 1)        # batch x 1 x [feat_dim+hidden]
                actual_embs[:, 0, :] = his_emb[mask].squeeze()  # history feats replace bos embedding
                decode_input = torch.cat([guid, actual_embs], 2)  # batch x [max_sent_len-1] x [emd_dim + feat_dim]  
                hidden = self.init_hidden(self.speaker_decoder, actual.size(0)).cuda() 
                actual_out = dynamicRNN(self.speaker_decoder, decode_input, actual_lens, encoder=False, initialState= hidden)  # batch x batch_max_len x [em4d_dim + hidden]
                actual_out = self.rnn_drop_layer(actual_out)
                batch_len_max = actual_out.size()[1]
                actual_pred = self.word_classifier(actual_out)
                word_pred[mask, :batch_len_max] = actual_pred                              # -> (b, 6, 512)
                
            return word_pred
        else:
            vis_feat = vis_feat.unsqueeze(1)
            decode_input = vis_feat
            word_emb_drop = his_emb
            beamgener = Generator(self.embedder, self.speaker_decoder, self.word_classifier,
                                  self.eos, beam_size, max_caption_length= self.smax - 1)
            
            speak, score, state = beamgener.beam_search(word_emb_drop, decode_input)
            sent_token = speak[0]
            sent_pred = ' '.join([self.vocab[idx.item()] for idx in sent_token[:-1]])

            return sent_pred, sent_token