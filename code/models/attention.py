from torch import nn
import torch.nn.functional as F
import torch
import math

# Attention mechanism module

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size))
            stdv = 1. / math.sqrt(self.v.size(0))
            self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.batch_score(hidden, encoder_outputs)
        att_weight = F.softmax(attn_energies, 1)
        return att_weight
        
    def negforward(self, hidden, encoder_outputs):                  # inverse-attention mechanism
        attn_energies = self.batch_score(hidden, encoder_outputs)
        att_weight = F.softmax(-1 * attn_energies, 1)
        return att_weight

    def batch_score(self, hidden, encoder_outputs):
        if self.method == 'dot':
            encoder_outputs = encoder_outputs.permute(1, 2, 0)
            energy = torch.bmm(hidden.transpose(0, 1), encoder_outputs).squeeze(1)
        elif self.method == 'general':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            energy = self.attn(encoder_outputs.contiguous().view(-1, self.hidden_size)).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(hidden.transpose(0, 1), energy.permute(1, 2, 0)).squeeze(1)
        elif self.method == 'concat':
            length = encoder_outputs.size(0)
            batch_size = encoder_outputs.size(1)
            attn_input = torch.cat((hidden.repeat(length, 1, 1), encoder_outputs), dim=2)
            energy = F.tanh(self.attn(attn_input.view(-1, 2 * self.hidden_size)).view(length, batch_size, self.hidden_size))
            energy = torch.bmm(self.v.repeat(batch_size, 1, 1), energy.permute(1, 2, 0)).squeeze(1)
        return energy