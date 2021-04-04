from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
from .model_utils import maskedCEL

class MaxMarginLoss(nn.Module):
    def __init__(self, margin, lama, lamb):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin
        self.lama = lama
        self.lamb = lamb
    def forward(self, pos_pos_pred, neg_pos_pred, pos_neg_pred, target_sents, weight = None):
        '''compute loss'''
        round_max = target_sents.size()[1]
        mask = target_sents.data.gt(0)
        nonzero = Variable(torch.nonzero(mask)).cuda()
        word_num = len(nonzero)
        pos_pos_loss = 0
        neg_pos_loss = 0
        pos_neg_loss = 0
        for j in range(round_max):
            pos_pos_loss += maskedCEL(pos_pos_pred[:, j], target_sents[:, j])
            neg_pos_loss += maskedCEL(neg_pos_pred[:, j], target_sents[:, j])
            pos_neg_loss += maskedCEL(pos_neg_pred[:, j], target_sents[:, j])
        pos_pos_loss = pos_pos_loss / word_num
        neg_pos_loss = neg_pos_loss / word_num
        pos_neg_loss = pos_neg_loss / word_num
        loss = pos_pos_loss + self.lama * (max(0, self.margin + pos_pos_loss - neg_pos_loss)) + \
              self.lamb * (max(0, self.margin + pos_pos_loss - pos_neg_loss))
        return loss

    def reset(self):
        return 0
