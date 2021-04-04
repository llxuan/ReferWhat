import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def getSortedOrder(lens):

    sortedLen, fwdOrder = torch.sort(
        lens.contiguous().view(-1), dim=0, descending=True)
    _, bwdOrder = torch.sort(fwdOrder)
    if isinstance(sortedLen, Variable):
        sortedLen = sortedLen.data
    sortedLen = sortedLen.cpu().numpy().tolist()
    return sortedLen, fwdOrder, bwdOrder

def dynamicRNN(rnnModel,
               seqInput,
               seqLens,
               encoder = True,
               initialState=None,
               returnStates=False):
    '''
    RNN type: GRU
    to sove the unsort pad packed sequence
    '''
    sortedLen, fwdOrder, bwdOrder = getSortedOrder(seqLens)
    sortedSeqInput = seqInput.index_select(dim=0, index=fwdOrder)
    #print(sortedSeqInput.size())
    packedSeqInput = pack_padded_sequence(sortedSeqInput, lengths=sortedLen, batch_first=True)

    if initialState is not None:
        hx = initialState
        #assert hx[0].size(0) == rnnModel.num_layers  # Matching num_layers
    else:
        hx = None
    out, h_n = rnnModel(packedSeqInput, hx)
    rnn_output = h_n[-1].index_select(dim=0, index=bwdOrder)    # c_n

    if encoder is False:
        out = pad_packed_sequence(out, batch_first=True)
        rnn_output =out[0].index_select(dim=0, index=bwdOrder)

    if returnStates:
        h_n = h_n.index_select(dim=1, index=bwdOrder)
        return rnn_output, h_n
    else:
        return rnn_output

def maskedCEL(seq, gtSeq, reduce = True, mask = None, shift = True, reduce_method = 'sum'):
    # seq: b x r x s x vocab
    # gtSeq: b x r x s
    # mask: b x r x s
    # Shifting gtSeq 1 token left to remove <START>
    if shift:
        padColumn = gtSeq.data.new(gtSeq.size(0), 1).fill_(0)
        target = torch.cat([gtSeq, padColumn], dim=-1)[:, 1:]   # shift
    else:
        target = gtSeq
    target = target.contiguous().view(-1, 1).squeeze()
    if mask is None:
        mask = torch.ge(target, 1)
    else:
        mask =mask.contiguous().view(-1, 1).squeeze()
    len_vocab = seq.size(-1)
    sent_pred = seq.contiguous().view(-1, len_vocab)
    target = torch.masked_select(target, mask=mask)
    nonzero = torch.nonzero(mask)

    dimension = len(nonzero)
    if dimension == 0:
        CE_loss = 0
    else:
        loss = nn.CrossEntropyLoss(reduce=False)
        sent_pred = sent_pred[nonzero.squeeze(1)]
        if reduce and reduce_method == 'sum':
            CE_loss = torch.sum(loss(sent_pred, target))
        elif reduce and reduce_method == 'avg':
            loss = nn.CrossEntropyLoss()
            CE_loss = loss(sent_pred, target)
        else:
            CE_loss = loss(sent_pred, target)
            
    return CE_loss

def acc_count(prob, target):
    mask = torch.ge(target, 1)
    target = target[mask]  # -> (? x 1)
    prob_pred = prob[mask]  # -> (? x 3 )
    nonzero = torch.nonzero(mask)
    total_num = len(nonzero)
    right_num = 0
    for i in range(total_num):
        tag_i = target[i]
        prob_i = prob_pred[i]
        _,pred_tag = torch.max(nn.functional.softmax(prob_i, dim = 0), dim = 0)
        if tag_i == pred_tag :
            right_num += 1

    return right_num, total_num