import torch.optim as optim 
import torch.nn as nn 
def loss_function(output, target):
    creterion = nn.CrossEntropyLoss(out,target)
    return creterion

def optimizer(net,type,lr):
    match type:
        case 'SGD':
            return optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        case 'Adam':
            return optim.Adam(net.parameters(), lr=lr))        
        case 'RMSprop':
            # define rh0 with initial value 0.9 in order to have a more robust training because past is not too much important
            return optim.RMSprop(net.parameters(), lr=lr,rho=0.9, eps=1e-08, weight_decay=0, momentum=0.9, centered=False)
            