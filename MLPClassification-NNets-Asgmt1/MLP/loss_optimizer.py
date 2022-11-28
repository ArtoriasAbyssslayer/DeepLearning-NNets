import torch.optim as optim 
import torch.nn.functional as F
def loss_function(output, target):
    creterion = F.cross_entropy(output,target)
    return creterion

def optimizer_select(net_params,type,lr):
        if type == 'SGD':
            optimizer = optim.SGD(net_params, lr=lr, momentum=0.9)
        elif type == 'Adam':
            optimizer = optim.Adam(net_params, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif type == 'Adadelta':
            optimizer = optim.Adadelta(net_params, lr=lr, rho=0.9, eps=1e-06, weight_decay=0)
        elif type == 'Rmsprop':
            optimizer = optim.RMSprop(net_params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        else:
            print('Optimizer not found')
        return optimizer
            