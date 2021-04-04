from __future__ import division

import logging
import math
import time
import os
import os.path as osp
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.nn.utils import clip_grad_value_
from models import MaxMarginLoss, MLP2
from tools import AverageMeter, select_optimizer,setup_logging
from tools import initializeWeights

class REGTrainer(object):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        self.lr = self.cfg.TRAIN.LR
        self.epoch = 1
        self.iter = 0
        self.opt_method = cfg.TRAIN.OPTIM
        self.save_root = os.path.join(cfg.SAVE_PATH, 'model_save', cfg.DATA.DATA_SET, 
                                      cfg.DATA.SPLIT, cfg.DATA.TYPE, cfg.DATA.FEATS_FILE,
                                      cfg.TRAIN.OPTIM + '_' + str(self.lr) + '_' +cfg.TRAIN.SAVE_NAME)
        print("model save path:", self.save_root)
        self.tbx_writer = SummaryWriter(log_dir=os.path.join(self.save_root,'tbx'))
        self.checkpoint_root = os.path.join(self.save_root, 'checkpoints')                       # save checkpoint and config path
        os.makedirs(self.checkpoint_root, exist_ok=True)
        self.config_root = os.path.join(self.save_root, 'config')
        os.makedirs(self.config_root, exist_ok=True)
        self.log_root = os.path.join(self.save_root, 'log')
        os.makedirs(self.log_root, exist_ok=True)
        self.dump_config()

        if cfg.RL_Train :
            self.append_init_RL()
        else:
            self.append_init_SL()

            
    
    def append_init_RL(self):
        self.mlp = MLP2(self.cfg.MODEL.RL).cuda()
        initializeWeights(self.mlp)
        print('append mlp initialize finish') 
        self.mse = nn.MSELoss(reduction='sum')
        self.rl_optim = select_optimizer(self.opt_method, params=self.model.parameters(), lr= self.lr)
        self.mlp_optim = select_optimizer(self.opt_method, params=self.mlp.parameters(), lr= self.lr)
       
    def append_init_SL(self):
        initializeWeights(self.model)
        print('append model initialize finish')   
        #self.apd_optim =select_optimizer(self.opt_method, params=self.model.parameters(), lr=self.lr)
        if self.cfg.TRAIN.OPTIM == 'SGD':
            self.apd_optim = torch.optim.SGD(self.model.parameters(), lr=self.lr,\
                momentum=self.cfg.TRAIN.MOMENTUM, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        elif self.cfg.TRAIN.OPTIM == 'Adam':
            self.apd_optim = torch.optim.Adam(self.model.parameters(), lr=self.lr,\
                betas=(0.8, 0.999), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
     

    def dump_config(self):
        self.cfg.dump_yaml(os.path.join(self.config_root, 'trainer_config.yml'))

    def lr_decay(self, optim):
        if self.cfg.MODEL.VIS_METHOD == 'visdif':
            if self.epoch % 1 == 0:
                if self.lr > self.cfg.TRAIN.MIN_LR:
                    for gId, group in enumerate(optim.param_groups):
                        optim.param_groups[gId]['lr'] *= self.cfg.TRAIN.LR_DECAY
                    self.lr *= self.cfg.TRAIN.LR_DECAY
                else:
                    self.lr = self.cfg.TRAIN.MIN_LR
            
        elif  self.cfg.MODEL.VIS_METHOD == 'atten' :
            if self.epoch % 1 == 0:
                if self.lr > self.cfg.TRAIN.MIN_LR:
                    for gId, group in enumerate(optim.param_groups):
                        optim.param_groups[gId]['lr'] *= self.cfg.TRAIN.LR_DECAY
                    self.lr *= self.cfg.TRAIN.LR_DECAY
                else:
                    self.lr = self.cfg.TRAIN.MIN_LR

       
    
    def _batch(self, data, training = True):
        '''
        :param data:
        :param training:
        :return:
        '''
        if training:
            self.model.train()
            if self.cfg.RL_Train:
                self.mlp.train()
            result = self.model(data)
        else:
            self.model.eval()
            if self.cfg.RL_Train:
                self.mlp.eval()
            with torch.no_grad():
                result = self.model(data)
        
        if self.cfg.RL_Train:
            (score, batch_cider, all_dec_out, all_pred_probs, pad_mask, cum_rewards, success_mask) = result
            mlp_result = self.mlp(all_dec_out.detach()).squeeze(3)
            mlp_result.data.masked_fill_(~pad_mask.data, 0)
            mlp_loss = self.mse(cum_rewards, mlp_result)
            batch_weight = cum_rewards - mlp_result.detach()
            weighted_ce_loss = - all_pred_probs * batch_weight * success_mask # only train successful example
            rl_loss = torch.sum(weighted_ce_loss, dim=2)
            rl_loss = torch.sum(rl_loss, dim=1)
            rl_loss = torch.mean(rl_loss, dim=0)
            result = (score , batch_cider, rl_loss, mlp_loss) 

        '''optimizer'''
        if training:
            # Backward propagation
            if self.cfg.RL_Train:
                self.rl_optim.zero_grad()
                self.mlp_optim.zero_grad()
                rl_loss = result[2]
                mlp_loss = result[3]
                rl_loss.backward()
                mlp_loss.backward()
            else:
                self.apd_optim.zero_grad()
                loss = result
                loss.backward()

            
            clip_grad_value_(self.model.hrnn.REGenc.parameters(), self.cfg.TRAIN.GC)
            clip_grad_value_(self.model.hrnn.speaker_decoder.parameters(), self.cfg.TRAIN.GC)  
            clip_grad_value_(self.model.hrnn.h_enc.parameters(), self.cfg.TRAIN.GC)

            # Backward propagation
            if self.cfg.RL_Train:
                self.rl_optim.step()
                self.mlp_optim.step()
            else:
                self.apd_optim.step()
            
        return result

    def _epoch(self, dataloader, training = True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        val1 = 0
        val2 = 0
        if self.cfg.RL_Train:
            batch_mlp_loss = AverageMeter()
            batch_rl_loss = AverageMeter()
            batch_score = AverageMeter()
            batch_cider = AverageMeter()
        else:
            perplexity = AverageMeter()
            batch_loss = AverageMeter()
            batch_score = AverageMeter()
            batch_cider = AverageMeter()


        for i, data in enumerate(dataloader):
            data_time.update(time.time() - end)
            self.iter += 1
            if self.cfg.RL_Train:
                (score , cider, rl_loss, mlp_loss) = self._batch(data, training=training)    
                batch_time.update(time.time() - end)
                end = time.time()
                batch_rl_loss.update(rl_loss.data)
                batch_mlp_loss.update(mlp_loss.data)
                batch_score.update(score)
                batch_cider.update(cider)
                if i % self.cfg.TRAIN.PRINT_PER_ITER == 0:
                    logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                 'MLP_loss {mlp_loss.val:.4f} ({mlp_loss.avg:.4f})\t'
                                 'RL_loss {rl_loss.val:.4f} ({rl_loss.avg:.4f})\t'
                                 'Score {score.val:.4f} ({score.avg:.4f})\t'
                                 'Cider {batch_cider.val:.4f} ({batch_cider.avg:.4f})'.format(
                        self.epoch, i, len(dataloader),
                        phase='TRAINING' if training else 'EVALUATING',
                        batch_time=batch_time,
                        data_time=data_time, mlp_loss=batch_mlp_loss, rl_loss=batch_rl_loss,
                        score = batch_score, batch_cider = batch_cider))
                    for param_group in self.rl_optim.param_groups:
                        logging.info('learning rate %f', param_group['lr'])
                    for param_group in self.mlp_optim.param_groups:
                        logging.info('learning rate %f', param_group['lr'])

                val1 = batch_score.avg
                val2 = batch_cider.avg
            else:
                loss = self._batch(data, training = training)
                batch_time.update(time.time() - end)
                end = time.time()
                batch_loss.update(loss.data)
                perplexity.update(math.exp(loss.data))
                
                if i % self.cfg.TRAIN.PRINT_PER_ITER  == 0:
                    logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                                 'Perplexity {perp.val:.4f} ({perp.avg:.4f})\t'
                                 'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        self.epoch, i, len(dataloader),
                        phase='TRAINING' if training else 'EVALUATING',
                        batch_time=batch_time,
                        data_time=data_time, perp=perplexity, loss=batch_loss))
                    for param_group in self.apd_optim.param_groups:
                        logging.info('learning rate %f', param_group['lr'])
                val1 = batch_loss.avg
                val2 = perplexity.avg
        return val1, val2

    def train_and_eval(self, train_loader, val_loader):
        setup_logging(os.path.join(self.log_root, 'log_train_and_eval.txt'))
        if self.cfg.RL_Train:
            max_score = 0.0
            max_cider = 0.0
            max_score_ckp = os.path.join(self.checkpoint_root, 'max_score_ckp')      # eval loss min ckp
            max_cider_ckp = os.path.join(self.checkpoint_root, 'max_cider_ckp')      # eval loss min ckp
            pretrain_model = os.path.join(self.save_root, '..', self.cfg.TRAIN.RL.pretrain_model)   # train longer and eval loss min ckp
            logging.info('\n resume training, load from ckp')
            check_point = torch.load(pretrain_model)
            self.model.load_state_dict(check_point['state_dict'])

        else:
            min_loss = 100
            min_loss_file = os.path.join(self.checkpoint_root, 'min_loss_ckp')     # eval loss min ckp
            ckp_file = os.path.join(self.checkpoint_root, '115_ckp')     # eval loss min ckp
        
        for i in range(1, self.cfg.TRAIN.MAX_EPOCH + 1):
            logging.debug("run arguments: %s", self.cfg)
            self.epoch = i
            val1, val2 = self._epoch(train_loader, training=True)
            eval_val1, eval_val2 = self._epoch(val_loader, training=False)
            
            if self.cfg.RL_Train:

                state_dict = {'epoch': self.epoch,
                            'iter': self.iter,
                            'state_dict': self.model.state_dict(),
                            'mlp_dict': self.mlp.state_dict(),
                            'rl_optim': self.rl_optim.state_dict(),
                            'mlp_optim': self.mlp_optim.state_dict()}

                if eval_val1 >= max_score - 0.0005:
                    if eval_val1 >= max_score:
                        max_score = eval_val1
                    torch.save(state_dict, max_score_ckp)
                if eval_val2 >= max_cider - 0.0005:
                    if eval_val2 >= max_cider: 
                        max_cider = eval_val2
                    torch.save(state_dict, max_cider_ckp)

                self.tbx_writer.add_scalar('_eval_model/socre', eval_val1, self.epoch)
                self.tbx_writer.add_scalar('_eval_model/cider', eval_val2, self.epoch)

                if self.cfg.RL_Train == False :
                    self.tbx_writer.add_scalar('_train_model/loss', val1, self.epoch)
                    self.tbx_writer.add_scalar('_train_model/perp', val2, self.epoch)
                    self.lr_decay(self.apd_optim)
                else:
                    self.tbx_writer.add_scalar('_train_model/socre', val1, self.epoch)
                    self.tbx_writer.add_scalar('_train_model/cider', val2, self.epoch)
            else:
                # var1 = loss, var2 = perp
                self.tbx_writer.add_scalar('_train_model/loss', val1, self.epoch)
                self.tbx_writer.add_scalar('_eval_model/loss', eval_val1, self.epoch)
                state_dict = {'epoch': self.epoch,
                              'iter': self.iter,
                              'state_dict': self.model.state_dict(),
                              'optim': self.apd_optim.state_dict()}
                if eval_val1 <= min_loss + 0.005:
                    if eval_val1 <= min_loss:
                        min_loss = eval_val1
                    torch.save(state_dict, min_loss_file)
                if self.epoch == 115:
                    torch.save(state_dict, ckp_file)
                self.lr_decay(self.apd_optim)

