# AutoEncoder Models
#### AutoEncoders on MNIST dataset
Demonstration AE:

IN:
![In](https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/in.gif)
OUT:
![Generated](https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/out.gif)


PCA denoise with 44 components


<img src="https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/Results/PCA_denoised.png" width="400" height="400"/>



This repository consists of varius types of encoding models in the hommonymous folder which are:
* [models/Autoencoder.py](https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/models/Autoencoder.py) := Simple AE model using simple reconstruction Loss
* [models/VAE.py](https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/models/VAE.py) := Linear Variational Autoencoder model using VI formulation i.e. KLD loss added to reconstruction loss
* [models/VAE_cnn.py](https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/models/VAE_cnn.py) := Convolutional Variational Autoencoder model
* [models/PCA.py](https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/models/PCA_denoise.py)

Also there are experiment scripts consist of::
* exps/train.py := universal model train script using cmd-line-args and a praser that provides info with -h
Example training:

``` python .\exps\train.py -m VAE -t 'bin_masking' -s VAE.ptr -hd 300 -L 85r ```

script usage:
```
python exps/train.py  -h
usage: train.py [-h] [--model_name MODEL_NAME] [--hidden_size HIDDEN_SIZE] [--latent_size LATENT_SIZE] [--data_masking DATA_MASKING] [--one_hot_labels]
                [--batch_size N] [--num_epochs N] [--optimizer N] [--save_name SAVE_NAME]

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME, -m MODEL_NAME
  --hidden_size HIDDEN_SIZE, -hd HIDDEN_SIZE
  --latent_size LATENT_SIZE, -L LATENT_SIZE
  --data_masking DATA_MASKING, -t DATA_MASKING
  --one_hot_labels, -oh
  --batch_size N, -b N  input batch size for training (default: 32)
  --num_epochs N, -e N  number of epochs to train (default: 100)
  --optimizer N         optimizer (default: Adam)
  --save_name SAVE_NAME, -s SAVE_NAME
```
* exps/evalution_inference.py := universal model test and call script to evaluate model on test dataset.

script usage:
``` 
$ python exps/evaluation_inference.py  -h
usage: evaluation_inference.py [-h] [--model_name MODEL_NAME] [--latent_size LATENT_SIZE] [--data_masking DATA_MASKING] [--batch_size BATCH_SIZE]
                               [--samples_num SAMPLES_NUM]

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME, -m MODEL_NAME
  --latent_size LATENT_SIZE, -L LATENT_SIZE
  --data_masking DATA_MASKING, -t DATA_MASKING
  --batch_size BATCH_SIZE, -b BATCH_SIZE
  --samples_num SAMPLES_NUM, -s SAMPLES_NUM  
 ```
 
 
 
 False generation
<img src="https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/Results/bin_masking_VAE_100epoch_reconstruction.png" width="400" height="400"/>

Good denoising example 
<img src="https://github.com/ArtoriasAbyssslayer/DeepLearning-NNets/blob/master/AutoEncoders/Results/PCA_denoised_154.png" width="400" height="400"/>

