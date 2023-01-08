import argparse
import sys
sys.path.append('..')
import torch 
import torchvision
import utils
import visualization_utils
from utils import adjust_learning_rate,clip_gradient,AverageMeter
from models import Autoencoder,VAE,VAE_cnn


#### GLOBAL PARAMTERES ####
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
ehidden_dim = 400
code_dim = 64
NUM_EPOCHS = 100
BATCH_SIZE = 32
INIT_LR_RATE = 3e-4
decay_lr_at = [80,100]
decay_lr_to = 3e-5
iterations = 100
n_classes = 10
grad_clip =  True


# Define the input placeholder
input_dim  = x_train.shape[1]
''' One Epoch Training Function '''
def train_epoch(dataloader,model,optimizer,criterion):
    input_size = len(dataloader.dataset)
    # Signal the model for training. This helps some specific layers such as BatchNorm,Dropout that operate
    # differently in training and evaluation/inference states
    model.train()
    # Initialize AveregeMeters

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter() # kld+recon losses buffer
    running_accuracies =  AverageMeter() # accuracy counting
    start_time = time.time()
    for batch_idx,(data,target) in enumerate(dataloader):
        data,target = data.to(DEVICE),target.to(DEVICE)
        data_time.update(time.time() - start)
        dec, mu, logs2 = model(data)
        # Add also the input transformation type to the loss function
        if(model.__name__=='Autoencoder'):
            sum_loss = model.rec_loss()
        else:
            kld_loss,recon_loss,sum_loss = model.recon_kld(dec,mu,logs2)
        # Clip gradients, if we observe this is needed
        if grad_clip is not None:
            clip_gradient(optimizer,grad_clip)
        # Init the gradients None -> +prefomance +lower memory footprint
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - start)
        # return loss per 100 batches
        if batch_idx%100 == 0:
            losses.update(sum_loss.item(),batch_idx*len(data))
            running_accuracies.update(accuracy,data)
            print('Epoch: [{0}][{1}/{2}]\t'
                'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch_num, i, len(train_loader),
                                                                batch_time=batch_time,
                                                                data_time=data_time, loss=losses))
        return loss


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
            if(model.__name__=='Autoencoder'):
                sum_loss = model.rec_loss()
                recon_losses.update(sum_loss.item())
            else:
                kld_loss,recon_loss,sum_loss = model.recon_kld(dec,mu,logs2)
                klds.update(kld_loss.item())
                recon_losses.update(recon_loss.item())
                losses.update(sum_loss.item())
    return klds.avg,recon_losses.avg,losses.avg
        
#TODO - FUNCTION BELLOW
'''Whole net train'''
def train(network,optimizer,criterion,dataloader,num_epochs):
    print("Training on {} device".format(DEVICE))
    network.to(DEVICE)
    global start_epoch,label_map,epoch,checkpoint,decay_lr_at
    param_buffer = None

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch,num_epoch))
        adfadf
        #

def main(args):
    checkpoint = args.model_name
    net = VAE(args.input_size,args.hidden_size,args.latent_size)
    optimizer = args.optimizer
    criterion = args.criterion
    config = {
        'batch_size':args.batch_size,
        'lr':INIT_LR_RATE,
        'data_masking': args.data_masking,
        'num_epochs': args.num_epochs,
        'optimizer':args.optimizer,
        'save_name':args.save_name,
        'model_name':args.model_name,
        'model':net,
        'latent_size':args.latent_size,
        'one_hot':args.one_hot_labels
        
    }
    train_loader,test_loader = utils.load_mnist(config.get('batch_size'),masking=config['data_masking'],one_hot_labels=['one_hot'],workers=4)
    kld_losses,recon_losses,train_losses = train(net,config['batch_size'],config['lr'],train_loader,test_loader,checkpoint)
    print(kld_losses)
    print(recon_losses)
    print(train_losses)

    plt.plot(kld_losses)
    plt.plot(recon_losses)
    plt.plot(train_losses)
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar=NameError(),
                        help='learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='Adam', metavar='N',
                        help='optimizer (default: Adam)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='N',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='N',
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--one_hot_labels',"-oh", action='store_true')
    parser.add_argument("--model_name","-m",default='VAE.py')
    parser.add_argument("--latent_size","-L",default=85,type=int)
    parser.add_argument("--data_masking","-t",default=None)
    parser.add_argument("--batch_size","-b",default=32,type=int)
    parser.add_argument("--num_epochs","-e",default=100,type=int)
    parser.add_argument("--save_name","-s",default='VAE.pt')
    arguments=parser.parse_args()
    main(arguments)