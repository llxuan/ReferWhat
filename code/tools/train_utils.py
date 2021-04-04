import logging
import subprocess
import torch
import numpy as np
import random
import math

def setup_seed(seed):
    """setting the random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return

def initializeWeights(root, itype='xavier'):
    assert itype == 'xavier', 'Only Xavier initialization supported'

    for module in root.modules():
        # Initialize weights
        name = type(module).__name__

        # If linear or embedding
        if name in ['Embedding', 'Linear']:
            print(name, 'weight initialize')
            fanIn = module.weight.data.size(0)
            fanOut = module.weight.data.size(1)

            factor = math.sqrt(2.0 / (fanIn + fanOut))
            weight = torch.randn(fanIn, fanOut) * factor
            module.weight.data.copy_(weight)
        elif name in ['LSTM', 'GRU']:
            print(name, 'weight initialize')
            for name, param in module.named_parameters():
                if 'bias' in name:
                    param.data.fill_(0.0)
                else:
                    fanIn = param.size(0)
                    fanOut = param.size(1)

                    factor = math.sqrt(2.0 / (fanIn + fanOut))
                    weight = torch.randn(fanIn, fanOut) * factor
                    param.data.copy_(weight)
        else:
            pass

        # Check for bias and reset
        if hasattr(module, 'bias') and type(module.bias) != bool:
            module.bias.data.fill_(0.0)

def setup_logging(log_file='log.txt'):
    """Setup logging configuration"""
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}

def select_optimizer(optimizer_name, params, lr):
    return __optimizers[optimizer_name](params, lr)

