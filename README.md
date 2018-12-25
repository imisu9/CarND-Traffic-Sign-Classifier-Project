# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/GTSRB_classes_in_order.jpg "GTSRB classes"
[image2]: ./examples/GTSRB_random_samples_in_order_of_classes.jpg "GTSRB random samples in the order of class"
[image3]: ./examples/training_data_set_distribution.png "Training data set distribution"
[image4]: ./examples/validation_data_set_distribution.png "Validation data set distribution"
[image5]: ./examples/test_data_set_distribution.png "Test data set distribution"
[image6]: ./examples/LeNet_architecture.png "LeNet architecture"
[image7]: ./examples/Sermanet_LeCun_architecture.jpg "Sermanet LeCun architecture"
[image8]: ./new_images/label_25.jpg "Label 25"
[image9]: ./new_images/label_33.jpg "Label 33"
[image10]: ./new_images/label_1.jpg "Label 1"
[image11]: ./new_images/label_22.jpg "Label 22"
[image12]: ./new_images/label_23.jpg "Label 23"
[image13]: ./new_images/label_13.jpg "Label 13"
[image14]: ./examples/prediction.png "Prediction Result "

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/imisu9/CarND-Traffic-Sign-Classifier-Project.git)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The data set is downloaded from [INI benchmark website](http://benchmark.ini.rub.de/?section=home&subsection=about). Two datasets are available: the German Traffic Sign Recognition Benchmark (GTSRB) and  the German Traffic Sign Detection Benchmark (GTSDB). These two has total of 43 classes.

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

![alt text][image1] ![alt text][image2]

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image3] ![alt text][image4] ![alt text][image5]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I reviewed [Traffic Sign Recognition with Multi-Scale Convolutional Networks by Pierre Sermanet and Yann LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Then I started researching on image pre-processing. 

It seemed that 
* Pre-processing is one of the most important step.
* It requires not just grayscaling but other techniques like also augmenting, rotating & centering.
* Lastly, normalization helps

So I went on to grayscale the images using `X_train = X_train.mean(axis=3, keepdims=True)`. Then I normalized it by using `X_train= cv2.normalize(X_train,  X_train, 0, 1, cv2.NORM_MINMAX)`. I've done the same process for validation data set and test data set.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model kept most of the original LeNet architecture.
![alt text][image6] 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscaled image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| dropout					|												|
| tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| local normalization					|												|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| dropout					|												|
| tanh					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| local normalization					|												|
| flatten					|	input 5x5x16, output 400						|
| Fully connected		| input 400 output 120						|
| dropout					|												|
| tanh					|												|
| Fully connected		| input 120 output 84						|
| dropout					|												|
| tanh					|												|
| Fully connected		| input 84 output n_classes						|
| Softmax				| etc.        									|

Hyper-parameters are set as:
* EPOCHS = 15
* BATCH_SIZE = 128
* LEARNING_RATE = 0.005
* KEEP_PROB_TRAIN = 0.75
* KEEP_PROB_VALID_TEST = 1.0

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* 1st trial: about 88% without updating anything: data preprocessing, hyperparameter, model architecture
* 2nd trial: data preprocessing: grayscale & normalization
    * about 90% with grayscale only
    * little better than grayscale only with adding normalization
* 3rd trial: peaked at about 93% with hyperparameter fine-tuning: increasing epoch to 15 & learning rate to 0.005
* 4th trial: activation function, dropout, batch normalization
    * first, changed relu to tanh which gave a peak of 95% accuracy.
    * seconddly, dropout affected validation accuracy southward. I set it to 0.75 to have it a minimum imapct, which gave similar accuracy.
    * thirdly, I considered batch normalization, which dropped the accuracy down by huge margin. To get to 90% accuracy, I had to raise # of epoches to 50. Insead, I switched to local_response_normalization which gave about one % bump to accuracy.
    * Lastly, applying Multi Stage architecuture which combines representation from multiple stages in the classifier to provide different scales of receptive fields to the classifier. Plus feature size of 108 - 200 was adopted from the paper. However, It did not helped much. My model did not go over 96% of accuracy.
    
I have tried the exact same model as described in the above paper using 108-200 feature size and the 2-lyaer classifier with 100 hidden units.
![alt text][image7]
It did not produce close to 99% accuracy. Rather it only gave me little over 90% accuracy after more than 20 epoches.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 96.1 
* test set accuracy of 92.8

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12] ![alt text][image13]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work      		| Road work   									| 
| Turn right ahead     			| Turn right ahead 										|
| Speed limit (30km/h)					| Speed limit (30km/h)											|
| Bumpy road	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|
| Yield			| Yield      							|

![alt text][image14]

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.76      		| Road work   									| 
| 1.00     			| Turn right ahead 										|
| 1.00					| Speed limit (30km/h)											|
| 1.00	      		| Bumpy Road					 				|
| 0.97			| Slippery Road      							|
| 1.00			| Yield      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?