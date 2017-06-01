Environment:

```
(tensorflow) [husam@husam-Z170 task4]$ python3
Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> tf.__version__
'1.1.0'
>>> import keras
Using TensorFlow backend.
>>> keras.__version__
'2.0.3'
>>> 

```

Running `pip3 freeze`:

```
(tensorflow) [husam@husam-Z170 task4]$ pip3 freeze
appdirs==1.4.3
apturl==0.5.2
audioread==2.1.4
bandmat==0.5
beautifulsoup4==4.4.1
bleach==2.0.0
blinker==1.3
Brlapi==0.6.4
bs4==0.0.1
chardet==2.3.0
checkbox-support==0.22
click==6.7
climate==0.4.6
command-not-found==0.3
cryptography==1.2.3
CVXcanon==0.1.1
cvxpy==0.4.8
cycler==0.10.0
Cython==0.25.1
decorator==4.0.11
defer==1.0.6
dill==0.2.6
downhill==0.4.0
ecos==2.0.4
entrypoints==0.2.2
fancyimpute==0.2.0
fastcache==1.0.2
feedparser==5.1.3
guacamole==0.9.2
h5py==2.7.0
html5lib==0.999999999
httplib2==0.9.2
idna==2.0
ipykernel==4.6.1
ipython==5.3.0
ipython-genutils==0.2.0
IPythonBell==0.9.9
ipywidgets==6.0.0
Jinja2==2.9.6
joblib==0.10.3
jsonschema==2.6.0
jupyter==1.0.0
jupyter-client==5.0.1
jupyter-console==5.1.0
jupyter-core==4.3.0
Keras==2.0.3
knnimpute==0.0.1
language-selector==0.1
librosa==0.4.3
louis==2.6.4
lxml==3.5.0
Mako==1.0.3
MarkupSafe==1.0
matplotlib==2.0.0
mistune==0.7.4
multiprocess==0.70.5
nbconvert==5.1.1
nbformat==4.3.0
notebook==5.0.0
notify2==0.3
numpy==1.12.1
oauthlib==1.0.3
onboard==1.2.0
packaging==16.8
padme==1.1.1
pandas==0.19.1
pandocfilters==1.4.1
pexpect==4.2.1
pickleshare==0.7.4
Pillow==3.1.2
plac==0.9.6
plainbox==0.25
prompt-toolkit==1.0.14
protobuf==3.2.0
ptyprocess==0.5.1
py==1.4.32
pyasn1==0.1.9
pycups==1.9.73
pycurl==7.43.0
Pygments==2.2.0
pygobject==3.20.0
PyJWT==1.3.0
pyparsing==2.2.0
PySocks==1.5.7
pytest==3.0.6
python-apt==1.1.0b1
python-dateutil==2.6.0
python-debian==0.1.27
python-systemd==231
pytz==2016.10
pyxdg==0.25
PyYAML==3.12
pyzmq==16.0.2
qtconsole==4.3.0
reportlab==3.3.0
requests==2.9.1
resampy==0.1.4
scikit-learn==0.18
scipy==0.19.0
screen-resolution-extra==0.0.0
scs==1.2.6
seaborn==0.7.1
sessioninstaller==0.0.0
simplegeneric==0.8.1
six==1.10.0
ssh-import-id==5.5
system-service==0.3
tensorflow-gpu==1.1.0
terminado==0.6
testpath==0.3
Theano==0.9.0
toolz==0.8.2
tornado==4.4.3
traitlets==4.3.2
twilio==5.6.0
ubuntu-drivers-common==0.0.0
ufw==0.35
unattended-upgrades==0.1
unity-scope-calculator==0.1
unity-scope-chromiumbookmarks==0.1
unity-scope-colourlovers==0.1
unity-scope-devhelp==0.1
unity-scope-firefoxbookmarks==0.1
unity-scope-gdrive==0.7
unity-scope-manpages==0.1
unity-scope-openclipart==0.1
unity-scope-texdoc==0.1
unity-scope-tomboy==0.1
unity-scope-virtualbox==0.1
unity-scope-yelp==0.1
unity-scope-zotero==0.1
unity-tweak-tool==0.0.7
urllib3==1.13.1
usb-creator==0.3.0
virtualenv==15.0.1
wcwidth==0.1.7
webencodings==0.5.1
Werkzeug==0.12.1
widgetsnbextension==2.0.0
xdiagnose==3.8.4.1
xgboost==0.6
xkit==0.0.0
XlsxWriter==0.7.3
youtube-dl==2017.4.2
(tensorflow) [husam@husam-Z170 task4]$ 

```

-------------------


# ORIGINAL TASK INSTRUCTIONS

All the tasks are to be completed using the [keras Sequential interface](https://keras.io/getting-started/sequential-model-guide/). You can but are not required to use the scikit-learn wrappers included in keras. We recommend to run all but Task 1 on the habanero cluster which provides GPU support (though you could also run task 2 on a CPU). Running on your own machine without a GPU will take significantly longer. We have limited access to GPU resources on the cluster, so make sure to start working on the homework well in advance. Feel free to use any other compute resources at your disposal.

You can find instructions on accessing the cluster below.
Task 1 [10 Points]
Run a multilayer perceptron (feed forward neural network) with two hidden layers and rectified linear nonlinearities on the iris dataset using the keras [Sequential interface](https://keras.io/getting-started/sequential-model-guide/). Include code for model selection and evaluation on an independent test-set.
[4pts for running model, 3pts for correct architecture, 3pts for evaluation]
Task 2 [30 Points]
Train a multilayer perceptron on the MNIST dataset. Compare a “vanilla” model with a model Qusing drop-out. Visualize the learning curves.
[Running model 10pts, model selection 7.5pts, dropout 7.5pts, learning curve 5pts]

Task 3 [30 Points]
Train a convolutional neural network on the [SVHN dataset](http://ufldl.stanford.edu/housenumbers/) in format 2 (single digit classification). You should achieve at least 85% test-set accuracy with a base model. Also build a model using batch normalization. You can compare against other approaches reported [here](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#5356484e) if you’re curious. You are not required to use the unlabeled data. [10pts for working model, 15pts for 85%, 5pts for batch normalization, BONUS 10pts for acc >=90%]

Task 4 [30 points]
Load the weights of a pre-trained convolutional neural network, for example AlexNet or VGG, and use it as feature extraction method to train a linear model or MLP  on the pets dataset. You should achieve at least 70% accuracy. It’s recommended you store the features on disk so you don’t have to recompute them for model selection.
The pets dataset can be found [here](http://www.robots.ox.ac.uk/~vgg/data/pets/)
We will be working with the 37 class classification task.
[10pts for loading model, 10pts for retraining, 10pts for 70% accuracy]

Note: We have compiled a list of Keras tips to help you with problems that you may run into at the bottom.

Files to submit:
The directory should contain 4 folders, one for each task

README with the performance that you get on each task.
Code and the plots in the respective folders.

## Keras Tips

1. Make sure all your code is running on GPU. tf prints out statements like "I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcublas.so locally - See more at: http://www.nvidia.com/object/gpu-accelerated-applications-tensorflow-installation.html#sthash.bH1GFiGZ.dpuf" when CUDA is imported. Make sure you see this every time.
2. Preprocess the images before training a model. For the transfer learning task, make sure you do the same preprocessing as the model expects. 
3. VGG16 is built-in into Keras, so using it is a good idea. refer this: https://github.com/fchollet/deep-learning-models
4. Test your code on a small part of the data before training the model. You don't want your code to fail on a print statement after waiting for the network to train.
5. The training time for each task should not be more than 30 mins each. If you exceeding that, either you have too big of a model, or there is a mistake in your code, or it is running on CPU.
6. For task 3, make sure you are doing the reshape for the training set correctly. A direct reshape might give you garbled images. Write an image to the disk after reshaping to make sure that the input to the model is correct.
7. Please copy the scripts to submit jobs to your personal directory and modify them there.
8. Do not use k-fold cross validation.
9. The CUDA import statements doesn't seem to be a reliable way of testing whether your code is running on GPU. Use "sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))" instead. You can include this statement in the beginning of your code and it will print out what device is being used for execution every time.

