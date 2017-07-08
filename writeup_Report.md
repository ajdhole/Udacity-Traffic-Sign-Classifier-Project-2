# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./writeup_images/Input_data.png "Visualization"
[image2]: ./writeup_images/Input_data_viz.png "Image class plot"
[image3]: ./writeup_images/Processed_image.png "Processed image"
[image4a]: ./writeup_images/initial_image.png "initial Image"
[image4]: ./writeup_images/transform_image.png "transform Image"
[image5]: ./writeup_images/validation_curve.png "Validation Curve"
[image6]: ./writeup_images/test_images.png "Test Images"
[image7]: ./writeup_images/detected_images.png "Top 3 Detected Images"
[image8]: ./writeup_images/images_sofmax.png "Softmax Probability"
[image9]: ./writeup_images/Featuremap1.png "Featuremap Conv 1"
[image10]: ./writeup_images/Featuremap2.png "Featuremap Conv 1_pool"
[image11]: ./writeup_images/Featuremap3.png "Featuremap Conv 2"
[image12]: ./writeup_images/Featuremap4.png "Featuremap Conv 2_pooling"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  This is a summary.  Continue further to read each in depth.   
1. As per project requirements, Traffic Sign Classifier project Notebook, HTML file and write up Files submitted.
2. Dataset summary & image visualization with image type distribution plot created.
3. Design & test model: which includes preprocessing, model architecture, hyperparameter tunning, training, and solution. 
4. Test model on new images, Trained model tested on new image downloaded from web, and plotted its softmax probability. 
5. Featuremap visualisation, Featuremap for all convolution layaer and pooling layer plotted.

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

#### 1a. Here is a link to my [project code](https://github.com/ajdhole/Traffic_Sign_Classisier/blob/master/Traffic_Sign_Classifier-Copy1.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set.
Initial training data charecteristics is shown below:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. Below images shows the German Traffic Signs and its class. Image plot showing the number of images per class.

![alt text][image1] 

Below image shows the trainng data distributution per class

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* As a first step, To generalise model better, I have decide to increase number of images and adding verity of image quality like, rotated image(As like seeing image from angular direction), translated image, share image, to achive this I have used transform_image function
which operates on image to give Image Rotation, Translaton and Share.

Here is an example of a traffic sign image before and after transform_image.

![alt text][image4a] ![alt text][image4]


* As a Second step in image processing, I decided to convert the images to grayscale because we are expecting network to detect traffic sign by geometry and shape and not by color, so to avoid wrong learning of network(i.e. sign detection by color), I have converted images into grayscale.

* As a last step, I normalized the image data because its always better to have image data with zero mean and equal variance
if its not case then it will be difficult for optimizer to find solution, well conditioned data i.e. zero mean and equal variance makes it lot easier to find optimum solution, image data comes in 0 to 255 pixle value, to normalize it I have used (X-Xmean)/std.

Here is an example of an transform image and image after grayscale and normalization:

![alt text][image4]![alt text][image3]

The difference between the original data set and the augmented data set is the following ...

**Existing data

* Number of training examples = 34799
* Number of Validation examples = 4410
* Number of testing examples = 12630

**Data after Augmentation

* Number of training examples after image augment = 86430
* Number of testing examples after image augment = 12630

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                                  |     Description                                                                  |
|:--------------------------:|:------------------------------------------------------:|
| Input                                  | 32x32x1 Grayscale image                                          |
| Convolution_1 5x5       | 1x1 stride, VALID padding, outputs 28X28X6                |
| RELU                                  |                                                                                              |
|Dropout                             |Keep probability = 0.7                       |
| Max pooling                    | 2x2 stride,  outputs 14x14x6                                    |
| Convolution_2 5x5       | 1x1 stride, VALID padding, outputs 10x10x16   |
| RELU                                  |                                                                                              |
| Max pooling                    | 2x2 stride,  outputs 5x5x16                                       |
| Fully connected_0        | Output = 400.                                                                 |
|Dropout                             |Keep probability – 0.6                                                 |
| Fully connected_1        | Output = 120.                                                                 |
| RELU                                  |                                                                                              |
| Fully connected_2        | Output = 84.                                                                   |
| RELU                                  |                                                                                              |
|Dropout                             |Keep probability = 0.6                                                 |
| Fully connected_3        | Output = 43.                                                                   |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, Initially I used same Lenet Modle as given in class tutorial, but validation accuracy of model was very low so To optimize it I have tried various approach to increase accuracy like adding droupout at fully connected layer, also changing keep probability value, additon of dropout at convolution 1 layer with different keep probability also ADAM optimizer gives better accuracy than SGD optimizer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

To acehive better validation accuracy I worked out on various hyperparameter like learning rate, epoch's, dropout keep probability and its position in network.

#### Following parameter value gives best result for my network:

* Learning Rate = 0.0009
* Epoch = 70
* batch Size = 100
* Dropout at FC 0 layer = 0.6
* Dropout at FC 2 layer = 0.6
* Dropout at Conv 1 layer = 0.7

#### Learning from various experiments: 

* Learning rate between 0.001 and 0.0009 gives better accuracy.
* Epoch value not fixed, it should be optimized based on model and various experiments.
* Batch Size - Higher batch size train model faster but need higher computation memory. Lower batch size takes more time to train model but works on lower computation power.
* 


The code for calculating the accuracy of the model is located in the 21th and 23rd cell of the Ipython notebook.

### My final model results are:

* training set accuracy of ----: 99.9%
* validation set accuracy of --: 99.6%
* test set accuracy of---------: 92.3%

Below graph shows validation accuracy over number of Epoch's.

 ![alt text][image5] 

If an iterative approach was chosen:

#### What was the first architecture that was tried and why was it chosen?

   * I  have used LeNet architecture as suggested in class tutorial.
   
#### What were some problems with the initial architecture?

   * While working on LeNet architecture I have notice that accuracy of model not yielding enough due to image data was not pre-processed also the model overfitting data which result in poor validationn accuracy.
   
#### How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

  * Initial model was overfitting data and not yielding required accuracy for validation data set, so to avoid this we have introduced dropout at first convolution layer, at first Fully connected layer and at last Fully connected layer, keep probability for Fully conected layer was same i.e. 0.6, but for first convolutional layer it was quite high 0.7 to avoid underfitting.
  
#### Which parameters were tuned? How were they adjusted and why?

  * Learning Rate, Batch Size, Epoch, Keep probability for dropout.
   * Learning Rate:- Higher Learning rate train model faster but stagnant earlier than acheving its full potential, whereas for lower learning rate model train slower but it achieves lowest possible loss for that model. in our model learning rate of 0.0009 yields better results.
   * Batch Size:- Since we can not train whole model at once due to computation power limitation, so we split model in batches, calculate all parameter for each batches and cascade it to top level for complete model, ats again on model size we can deside batch size, in our model Batch size is 128.
   * Keep Probability:- To avoid overfitting in model we have to introduced dropout at different layer, In our model I have added dropout at Conv1 layer, at first fully connected layer, and at last fully connected layer but with different keep probability, like at earlier layer it is higher 0.7 in our case and at later layer it is 0.6.
  
   

### Test a Model on New Images

### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

  ![alt text][image6] 


The first image might be difficult to classify because image downloaded from web was not as per required size i.e 32x32x1 which is required size for Network, so we have converted image size and preprocessed it same as training data preprocessing.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image                                                |     Prediction                                |
|:----------------------------------------------------:|:---------------------------------------------:|
| Right-of-way at the next Intersection                | Right-of-way at the next Intersection         |
| Speed Limit(30km/h)                                  | Speed Limit(30km/h)                           |
| Priority Road                                        | Priority Road                                 |
| Keep Right                                           | Keep Right                                    |
|Turn Left Ahead                                       | Turn Left Ahead                               |
|General Caution                                       | General Caution                               |
|Road work                                             | Road work                                     |
| Stop                                  | Stop                          |


![alt text][image7]

Figure showing Top 3 Probabilities for detected images.

The model was able to correctly guess 8 of the 8 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The top five soft max probabilities is:

| Probability                                                           |     Prediction                                       |
|:---------------------------------------------------------------------:|:----------------------------------------------------:|
| 100%                                                                  | Right-of-way at the next Intersection                |
| 100%                                                                  | Speed Limit(30km/h)                                  |
| 100%                                                                  | Priority Road                                        |
| 100%                                                                  | Keep Right                                           |
|100%                                                                   | Turn Left Ahead                                      |
|100%                                                                   | General Caution                                      |
|81%                                                                    | Road work                                          |
| 86%                                                                  | Stop                               |


![alt text][image8]

Figure showing sofmax probability.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

 ### Visualizing feature map is the best technique to understand Convolutional network.
 
 ![alt text][image9]
  
  -Featuremap of conv1 Layer, the shape of this layer is 28x28x6 and this featuremap is for Right-of-way at the next Intersection sign class.
  
  ![alt text][image10]
    
  -Featuremap for Conv1 Pooling layer which is having output shape of 14x14x6.
  
  ![alt text][image11]
  
  -Featuremap for Conv2 layer of output shape of 10x10x16.
  
  ![alt text][image12]
  
  -Featuremap of Conv2 pooling layer which is having output shape of 5x5x16.
  
  

