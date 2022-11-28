### Training the models
Training the models could be done with both minimalTraining.py or train.py.The second is a more 
coprehensive and robust way to train the model so it is recommended.

Train.py also evaluates the model on random test set after training is finished


The train.py offers command line execution with the command
```console
    python.exe train.py [model_name]
```
[model_name]
model_name variable could take on of the below strings:
* SmallMLP
* NetworkBatchNorm
* DenseMLP

These models were created in the framework of this course and limited time.

The training process was made only for cifar10 at this point because there was no time to work on Intel
dataset and also iterators for Intel should be made