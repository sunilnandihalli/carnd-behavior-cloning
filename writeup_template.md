#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center.png "Center image"
[image3]: ./examples/left.png "Left Image"
[image4]: ./examples/right.png "Right Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode. This file also contains the model-creation code and hence is different from the one provided by default. This only reads the weights from the model-file.
* model_weights.h5 containing a trained convolution neural network weights only. 
* data_to_sstable.py to preprocess the data and store as tfrecords in an sstable for efficient iteration during training time.
I built the model on our cluster which unfortunately only supported python2 and I had trouble reading the model file directly into the drive.py which was running on python3 on my laptop. I used a hack to recreate the model using code in drive.py and simply load the saved weights. 

* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_weights.hd5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the weights of the CNN. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 20 and 50 (model.py lines 95-111) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 100). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 108). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 311-316). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 111).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach
The overall solution strategy was to save the data to a sstable as that would enable efficient access to data. I also wanted to save on the effort to convert jpg to byte-arrays everytime an example is generated. In the end I don't think this mattered.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because this was suggested.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added a layer of Dropout with probability of 0.5.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I augmented the data with left and right camera images with slight correction to the steering angles. I also drove the vehicle in the opposite direction to get more data. I drove the vehicle along the left and right sides of the road to be able to teach model to steer when close to the edge of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 95-112) consisted of a convolution neural network with the following layers and layer sizes ...
* Lambda-layer for normalization
* Cropping2D to remove unnecessary region of the fov of the camera
* Conv2D of filter size 5x5 depth 20 and a relu activation
* MaxPooling with strides 2x2
* Conv2d of filter size 5x5 depth 50 and a relu activation
* MaxPoolint with strides of 2x2
* Fully Connected layer with 240 nodes
* Dropout layer with rate 0.5
* Fully Connected layer with 84 nodes
* Fully Connected layer with 1 node


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer well when near the edges of the road.

![alt text][image3]
![alt text][image4]

Then I repeated this process on track two in order to get more data points.

After the collection process, I had about 77000 number of data points. I then preprocessed this data by converting them TFExamples and stored the shuffled data as an SSTable for efficient access.

The data in the SSTable was already shuffled. I just took care of the splitting into train and validation data in the generators by setting appropriate offsets. I used 10% of the data as validation data.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
