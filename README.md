Project code for Udacity's AI Programming with Python Nanodegree program. 
In this project, code developed for an image classifier built with PyTorch, 
then converted into a command line applications: train.py, predict.py.

The image classifier to recognize different species of flowers. Dataset contains 102 flower categories.

In Image Classifier Project, Densenet from torchvision.models pretrained models was used. 
It was loaded as a pre-trained network, based on which defined a new, untrained feed-forward network as a classifier, 
using ReLU activations and dropout. Trained the classifier layers using backpropagation using 
the pre-trained network to get the features. The loss and accuracy on the validation set were tracked to 
determine the best hyperparameters.

Command line applications train.py and predict.py

For command line applications there is an option to select either VGG13, VGG16_bn, or Densenet models.

Following arguments are mandatory or optional for train.py

'data_dir'. 'Provide data directory. Mandatory argument', type = str
'--save_dir'. 'Provide saving directory. Optional argument', type = str
'--arch'. 'vgg13', 'vgg16_bn', or 'densenet121' will be used', type = str
'--learning_rate'. 'Learning rate, default value 0.001', type = float
'--hidden_units'. 'Hidden units in Classifier. Default value is 512, type = int
'--epochs'. 'Number of epochs'. Default value is 3, type = int
'--GPU'. "Option to use GPU", type = str

Following arguments are mandatory or optional for predict.py

'load_img'. 'Provide path to image. Mandatory argument', type = str
'pil_img'. 'Provide path to checkpoint. Mandatory argument', type = str
'--top_k'. 'Top K most likely classes. Optional', type = int
'--cat_file'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str
'--gpu'. "Option to use GPU. Optional", type = str
