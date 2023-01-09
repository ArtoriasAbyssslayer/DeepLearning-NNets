import argparse
import sys
import os 
sys.path.append(os.getcwd())
import torch 
import torch.optim as optim
import utils
import visualization_utils
from utils import adjust_learning_rate,clip_gradient,AverageMeter
from models import Autoencoder,VAE,VAE_cnn
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import time


#### GLOBAL PARAMETERS ####
cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Default Vals VAE
# INPUT_DIM = 784
# ehidden_dim = 400
# code_dim = 64
# BATCH_SIZE = 32
INIT_LR_RATE = 3e-4
decay_lr_at = [80,100]
decay_lr_to = 0.2
# For SDG or RMSProp Optim
momentum = 0.9
rho = 0.9
weight_decay = 3e-5
iterations = 100
n_classes = 10
grad_clip =  True

''' One Epoch Training Function '''
def train_epoch(dataloader,model,optimizer):
    # Signal the model for training. This helps some specific layers such as BatchNorm,Dropout that operate
    # differently in training and evaluation/inference states
    model.train()
    # Initialize AveregeMeters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter() # kld+recon losses buffer
    running_accuracies =  AverageMeter() # accuracy counting
    start_time = time.time()
    batch_idx = 0
    for batch_idx,(data,target) in enumerate(dataloader):
        data,target = data.to(DEVICE),target.to(DEVICE)
        data_time.update(time.time() - start_time)
        dec, mu, logs2 = model(data)
        # Add also the input transformation type to the loss function
        if(model.__class__.__name__=='Autoencoder'):
            sum_loss = model.rec_loss()
        else:
            kld_loss,recon_loss,sum_loss = model.loss_function(dec,data,mu,logvar=logs2)
        # Clip gradients, if we observe this is needed
        if grad_clip is not None:
            clip_gradient(optimizer,grad_clip)
        # Init the gradients None -> +prefomance +lower memory footprint
        optimizer.zero_grad()
        sum_loss.backward()
        optimizer.step()
        batch_time.update(time.time() - start_time)
        
        # return loss per 100 batches
        if batch_idx%100 == 0:
            losses.update(sum_loss.item())
            batch_current =  batch_idx*len(data)
            
            print(f"Batch Progress: [{batch_current}/{len(dataloader.dataset)}]\t" 
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(batch_time=batch_time,
                                                                data_time=data_time, loss=losses))
    
    return sum_loss,losses

'''Test dataset one epoch eval'''
def test_epoch(dataloader,model):
    batches_num = len(dataloader)
    model.eval()
    losses = AverageMeter()
    klds = AverageMeter()
    recon_losses = AverageMeter()
    # Use torch no grad and model.eval() to set model in not training state
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(dataloader):
            data,target = data.to(DEVICE),target.to(DEVICE)
            dec, mu, logs2 = model(data)
            if(model.__class__.__name__=='Autoencoder'):
                sum_loss = model.rec_loss()
                recon_losses.update(sum_loss.item())
            else:
                kld_loss,recon_loss,sum_loss = model.loss_function(dec,data,mu,logvar=logs2)
                klds.update(kld_loss.item())
                recon_losses.update(recon_loss.item())
                losses.update(sum_loss.item())
    return klds.avg,recon_losses.avg,losses.avg
        

'''Whole net train'''
def train(network,batch_size,lr,trainloader,testloader,min_loss,optimizer,start_epoch,num_epochs,save_name):
    # moce  network to selected device
    print("Training on {} device".format(DEVICE))
    network.to(DEVICE)
    # Create Losses Buf
    recon_losses_buf = np.zeros((num_epochs),)
    kld_losses_buf = np.zeros((num_epochs),)
    losses_buf = np.zeros((num_epochs),)
   
    global checkpoint,decay_lr_at
    # train
    print("--------------Model Training Initialized-----------------")
    for epoch in tqdm(range(start_epoch,num_epochs)):
        print("Epoch {}\n----------".format(epoch))
    
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)
        loss,losses = train_epoch(trainloader,network,optimizer)
        # print('\nTrain: Average loss: {:.4f}'.format(losses.avg))
        # if(epoch%10==0): I would  do it every 10 epochs decided to do it every epoch
    
        kld,recon,loss=test_epoch(testloader, network)
        print('\nTest set: Average loss: {:.4f}, KLD Average loss:{:4f}, Recon Average loss:{:4f}\n'.format(loss,kld,recon))
        kld_losses_buf[epoch] = kld
        recon_losses_buf[epoch] = recon
        losses_buf[epoch] = loss

        # Save model checkpoint  if it got better    
        if min_loss is None or  loss < min_loss:    
            utils.save_model(network,min_loss,optimizer,save_name) 
            print('TRAINING FINISHED OPTIMAL MODEL SAVED')
        # Save Net as checkpoint if interaption
        utils.save_checkpoint(epoch, min_loss,network, optimizer,save_name=save_name)
        print('CURRENT MODEL CHECKPOINT SAVED')
    return kld_losses_buf,recon_losses_buf,losses_buf
def net_builder(config):
    if config['model_name'] == 'Autoencoder':
        net = Autoencoder.AutoEncoder(input_dim = 784,hidden_dim=config['hidden_size'],latent_dim=config['latent_size'],gaussian_blurred_input=config['data_masking'])
    elif config['model_name'] == 'VAE':
        if config['data_masking'] == 'bin-masking':
            net = VAE.VAE(input_dim = 784,e_hidden_dim=config['hidden_size'],latent_dim=config['latent_size'],bernoulli_input=config['data_masking'],gaussian_blurred_input=None)
        else:
            net = VAE.VAE(input_dim = 784,e_hidden_dim=config['hidden_size'],latent_dim=20,bernoulli_input=None,gaussian_blurred_input=None)
    elif config['model_name'] == 'PCA':
        net = utils.load_pca_model(model, config['model_name'])
    else:   # model_name = 'VAE_cnn'
        if config['data_masking'] == 'bin-masking':
            net = VAE_cnn.VAE_cnn(config['latent_size'],bernoulli=True)
        else:
            net = VAE_cnn.VAE_cnn(config['latent_size'],bernoulli=False)
    return net
def main(args):
    global start_epoch,epoch,checkpoint,decay_lr_at,decay_lr_to
    start_epoch = 0
    config = {
        'batch_size':args.batch_size,
        'lr':INIT_LR_RATE,
        'data_masking': args.data_masking,
        'num_epochs': args.num_epochs,
        'optimizer':args.optimizer,
        'save_name':args.save_name,
        'model_name':args.model_name,
        'latent_size':args.latent_size,
        'one_hot':args.one_hot_labels,
        'hidden_size':args.hidden_size
    }
    checkpoint = config['save_name']
    net = net_builder(config)
    min_loss = 0
    if checkpoint is None:
        config['save_name'] = config['model_name']
    else:
        if os.path.exists(checkpoint):
            param_buffer = torch.load(checkpoint)
            model_state, optimizer_state = param_buffer['model_state_dict'],param_buffer['optimizer_state_dict']
            # Load pretrained weights and state
            net = net.load_state_dict(model_state)
            optimizer = optimizer.load_state_dict[optimizer_state]
            start_epoch = checkpoint['epoch'] + 1
            print('\n Loaded Checkpoint from epoch %d.\n'.format(start_epoch))
            min_loss = prams_buffer['test_loss'] if param_buffer != None else 0
    biases = list()
    not_biases = list()
    for param_name, param in net.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    # Select optimizer if model is not from checkpoint
    if config['optimizer']=='Adam':
        optimizer = optim.Adam(params=[{'params': biases, 'lr': 10 * config['lr']}, {'params': not_biases}],lr=config['lr'],weight_decay=weight_decay)
    elif config['optimizer']=='SGD':
        optimizer = optim.SGD(params=[{'params': biases, 'lr': 10 * config['lr']}, {'params': not_biases}],lr=config['lr'], momentum=momentum,weight_decay=weight_decay)
    elif config['optimizer']=='RMSProp':
        optimizer = optim.RMSProp(params=[{'params': biases, 'lr': 10 * config['lr']}, {'params': not_biases}],lr=config['lr'],alpha=rho,momentum=momentum,weight_decay=weight_decay)
    else:
        print("Optimizer not supported")
    
    '''print model summary'''
    print(net)
    # Train the model)
    '''load mnist'''
    train_loader,test_loader = utils.load_mnist(config.get('batch_size'),masking=config['data_masking'],one_hot_labels=False,workers=4)
    '''train and eval model'''
    print("Starting from epoch:{}".format(start_epoch))
    training_toc = time.time()
    kld_losses,recon_losses,train_losses = train(net,config['batch_size'],config['lr'],train_loader,test_loader,min_loss,optimizer,start_epoch,config['num_epochs'],config['save_name'])
    training_tic = time.time()
    print("---TRAINING FINISHED---\nElapsed Training Time {:.3f}s".format(training_tic-training_toc))
    '''print reconstruction,vi losses arrays'''
    print(kld_losses)
    print(recon_losses)
    print(train_losses)
    # Print curves
    visualization_utils.plot_loss_curves(kld_losses,recon_losses,train_losses,config['num_epochs'],config['model_name'],isTrain=True)
    print("End Of train Script")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name","-m",default='VAE')
    parser.add_argument("--hidden_size","-hd",default=400,type=int)
    parser.add_argument("--latent_size","-L",default=20,type=int)
    parser.add_argument("--data_masking","-t",default=None)
    parser.add_argument('--one_hot_labels',"-oh", action='store_true')
    parser.add_argument('--batch_size',"-b",type=int, default=32, metavar='N',help='input batch size for training (default: 32)')
    parser.add_argument('--num_epochs','-e', type=int, default=100, metavar='N',help='number of epochs to train (default: 100)')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='N',help='optimizer (default: Adam)')
    parser.add_argument("--save_name","-s",default=None)
    arguments=parser.parse_args()
    main(arguments)