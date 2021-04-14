# Slip Prediction
This project is an attempt to use a data drive, multi-modal, deep neural network to predict future tactile readings.

## Data Set
The dataset (email for access) is a set of kinestheticly demonstrations of robot motions with the robot gripping an object. The object was gripped with a two finger gripper, each finger with the 4x4 Xela Uskin tactile sensor.
The recored frequency was 48 frames a second.

## Data Set Formating
To format the data into sequences for training use file manual_data_models/format_manual_data.py. This script will save a set of numpy files containing all the relevant data for a time step sequence. These can then be loaded during training and validation time. To use the images set the hyper perameter "SAVE_IMAGES" to False. This will keep the dataset size down considerably.

## Visualisation of the dataset
within /manual_data_models/visualisation.py are a set of function to validate and view the dataset with different representations such as rescaled images and object based representations.

## Models
Models can be split into catagories by their representation type, here we have simple_models, conv_models and object_models. Each model has subtle or large changes. The models save training, validation and test data into numpy arrays for analysis.
