


## Task 3

The structure of the NN looked like this:
Conv layer 64 units,
Batch Norm,
relu,
Max Pooling

Conv layer 32 units,
Batch Norm,
relu,
Max Pooling

Conv layer 32 units,
Batch Norm,
relu,
Max Pooling

Flatten

Dense 128 units,
Batch Norm,
relu,
Droput .25

Dense 128 units,
Batch Norm,
relu,
Droput .50

Dense 128 units,
Batch Norm,
relu,
Droput .25

Dense 64 units,
Batch Norm,
relu,
Droput .50

Softmax 11 classes

The test error rate on an independent test set was 0.90285033804548254.

The loss/accuracy graph looks like this (10% of training set for validation):


![](/Homework-v/task3/loss_acc_curve.png?raw=true )






