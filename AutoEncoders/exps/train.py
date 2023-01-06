import argparse
import sys
sys.path.append('..')
import torch 
import torchvision
import utils
import visualization_utils
from utils import adjust_learning_rate,clip_gradient,AverageMeter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 784
ehidden_dim = 200
code_dim = 20
NUM_EPOCHS = 20
BATCH_SIZE = 32
LR_RATE = 3e-4
# Define the input placeholder
input_dim  = x_train.shape[1]
inputs = tf.placeholder(dtype=tf.float32,shape=(None,input_dim))

NUM_EPOCHS = 10
BATCH_SIZE = 128

# TODO DELETE BELLOW 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    autoencoder = AutoEncoder(input_dim,encoding_dim)
    for epoch in range(NUM_EPOCHS):
        for i in range(0,len(x_train),BATCH_SIZE):
            X = x_train[i:i+BATCH_SIZE]
            X_hat = autoencoder.forward(X)
            dX_hat = autoencoder.backward(X_hat)
            autoencoder.update(0.01)
        print("Epoch: {} Loss: {}".format(epoch,autoencoder.loss()))
    X_hat = autoencoder.forward(x_test)
    print("Test Loss: {}".format(autoencoder.loss()))
# Data Loading




''' One Epoch Training Function '''
def train_epoch(dataloader,model,optimizer,criterion):
    input_size = len(dataloader.dataset)
    # Signal the model for training. This helps some specific layers such as BatchNorm,Dropout that operate
    # differently in training and evaluation/inference states
    model.train()
    for batch_idx,(data,target) in enumerate(dataloader):
        data,target = data.to(DEVICE),target.to(DEVICE)
        
        dec, mu, logs2 = model(data)
        kld_loss
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,data)
        loss.backward()
        optimizer.step()
        # Clip gradients, if we observe this is needed
        if grad_clip is not None:
            clip_gradient(optimizer,grad_clip)

        if batch_idx % 10 == 0:
            #TODO add code for log batches
            pass

'''Whole net train'''
def train(model,optimizer,criterion,dataloader,num_epochs):
    # TODO
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name","-m",default='VAE.py')
    parser.add_argument("--latent_size","-L",default=85,type=int)
    parser.add_argument("--data_masking","-t",default=None)
    parser.add_argument("--batch_size","-b",default=32,type=int)
    parser.add_argument("--num_epochs","-e",default=100,type=int)
    parser.add_argument("--save_name","-s",default='VAE.pt')


   