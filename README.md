# Data-Science-Rice-Leaf-Disease-detection

##  Rice leaf disease detection by using Deep learning model Convolutional Neural Network(CNN).
## Introduction
The rice leaf suffers from several bacterial, viral, or fungal diseases and these diseases reduce rice production. Using a Dataset of Rice Leaf Disease images, a number of CNN-based models are trained to identify three common rice leaf diseases.

#### Convolutional Neural Network(CNN):
 A convolutional neural network (CNN) is a type of artificial neural network used in image recognition and processing that is specifically designed to process pixel data.
![image](https://user-images.githubusercontent.com/73145010/156520728-158283bb-d09d-40d2-b382-68f1ee2cd47f.png)


#### Dataset:
The dataset is Taken from Kaggle.

## Instructions for Running Python Notebooks Locally
- requiements.txt
-  Use google colab
- use jupyter nootbook

## I have divide the project into multiple steps
 * Importing Libraries
 * Data Augmentation & Split data Train,Test set: 
   In this section I augmented the data where I get variety of apporch in every single image after that I split data train & test set.
 * Model Building
   - input layer
   - convolution layer 1 (Conv1)
   - max pooling layer (Pooling1)
   - convolution layer 2 (Conv2)
   - max pooling layer 2 (Polling2)
   - convolution layer 3 (Conv3)
   - Three dense layers (Dense1, Dense2 and Dense3) and
   - an output (softmax) layer

 * Model Evaluate 
   - validation loss: 0.020420538261532784
 
    ![image](https://user-images.githubusercontent.com/73145010/156523583-26084f6f-0dc2-49a8-8e6a-6df0504df634.png)
   - validation accuracy 1.0
  
    ![image](https://user-images.githubusercontent.com/73145010/156523496-db84b873-a4ab-42d5-8653-ffca9d8f5382.png)
   
   
 * Make Prediction: This section for take image from test section and predict there label.

 * Single image Prediction: 
  In this section you can upload any image and can be check is it the problem of leaf Bactorial leaf blight, Brown spot or Leaf smut.
 


