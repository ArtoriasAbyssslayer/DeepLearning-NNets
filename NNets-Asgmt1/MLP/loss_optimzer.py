import torch.optim as optim 
import torch.nn as nn 
def loss_function(output, target):
    creterion = nn.CrossEntropyLoss(out,target)
    return creterion

def optimizer_select(net_params,type,lr):
    match type:
        case 'SGD':
            return optim.SGD(net_params, lr=lr, momentum=0.9)
        case 'Adam':
            return optim.Adam(net_params, lr=lr))        
        case 'RMSprop':
            # define rh0 with initial value 0.9 in order to have a more robust training because past is not too much important
            return optim.RMSprop(net_params, lr=lr,rho=0.9, eps=1e-08, weight_decay=0, momentum=0.9, centered=False)
            