<p>Project code for Udacity's AI Programming with Python Nanodegree program. 
In this project, code developed for an image classifier built with PyTorch, 
then converted into a command line applications: train.py, predict.py.</p>

<p>The image classifier to recognize different species of flowers. Dataset contains 102 flower categories.</p>

<p>In Image Classifier Project, Vgg13, Vgg16_bn, and Densenet from torchvision.models pretrained models was used. 
It was loaded as a pre-trained network, based on which defined a new, untrained feed-forward network as a classifier, 
using ReLU activations and dropout. Trained the classifier layers using backpropagation using 
the pre-trained network to get the features. The loss and accuracy on the validation set were tracked to 
determine the best hyperparameters.</p>

<p>Command line applications train.py and predict.py</p>

<p>For command line applications there is an option to select either VGG13, VGG16_bn, or Densenet models.</p>

<p>Following arguments are mandatory or optional for train.py: </p>

'--data_dir'. 'Provide data directory. Mandatory argument', type = str <br />
'--save_dir'. 'Provide saving directory. Optional argument', type = str <br />
'--arch'. 'vgg13', 'vgg16_bn', or 'densenet121' will be used', type = str <br />
'--learning_rate'. 'Learning rate, default value 0.001', type = float <br />
'--hidden_units'. 'Hidden units in Classifier. Default value is 512, type = int <br />
'--epochs'. 'Number of epochs'. Default value is 3, type = int <br />
'--GPU'. "Option to use GPU", type = str <br />

<p>Following arguments are mandatory or optional for predict.py: </p>

'--load_img'. 'Provide path to image. Mandatory argument', type = str <br />
'--pil_img'. 'Provide path to checkpoint. Mandatory argument', type = str <br />
'--top_k'. 'Top K most likely classes. Optional', type = int <br />
'--cat_file'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str <br />
'--gpu'. "Option to use GPU. Optional", type = str <br />
