import torch 
from torch import nn
import torch.nn.utils.prune as prune


def compute_sparsity_toy(model):

    print('Null values:')
    print(torch.sum(model.W == 0))
    print('Total values:')
    print(model.W.nelement())

    s = 100. * float(torch.sum(model.W == 0)) / float(model.W.nelement())

    return s

def compute_sparsity_global(model):
        s = 100. * (float(sum([torch.sum(module.weight == 0) for module in model.modules() if isinstance(module, nn.Parameter)])) / float(sum([module.weight.nelement() for module in model.modules() if isinstance(module, nn.Parameter)])))
        return s
    
def prune_model(model):

    module = model

    with torch.no_grad():
        model = prune.l1_unstructured(module, name="W", amount=0.1)

    return model