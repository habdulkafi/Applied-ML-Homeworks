## Task 4
I loaded the VGG model and extracted the features, then saved them to `features.npz`.  If you want to rerun that (because I can't upload a giant file to github), set the `reload` variable to `True`

The layout of the NN looks like:

Dense 256 units, Batch Norm, relu

Dense 256 units, Batch Norm, relu

Dense 64 units, Batch Norm, relu

Dense 32 units, Batch Norm, relu

Softmax 37 classes

The accuracy on an independent test set was 0.89580514208389717.  

The accuracy/loss curve looks like (the validation scores are from 20% of the training data):

![](/task4/pets-curve.png?raw=true )
