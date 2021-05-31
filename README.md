# About
Project from Udacity. <br>
Final project from 1st Section 
Original [GitHub project](https://github.com/udacity/P1_Facial_Keypoints)

# Instructions

## Submission Files
* :white_check_mark: models.py: 
* :white_check_mark: Notebooks 2: Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints 
* :white_check_mark: Notebooks 3: Facial Keypoint Detection Using Haar Cascades and your Trained CNN

## Project Rubric

### `models.py`

#### Specify the CNN architecture
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark: Define a CNN in `models.py`. |  Define a convolutional neural network with at least one convolutional layer, i.e. self.conv1 = nn.Conv2d(1, 32, 5). The network should take in a grayscale, square image. |


### Notebook 2

#### Define the data transform for training and test data
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark: Define a `data_transform` and apply it whenever you instantiate a DataLoader. |  The composed transform should include: rescaling/cropping, normalization, and turning input images into torch Tensors. The transform should turn any input image into a normalized, square, grayscale image and then a Tensor for your model to take it as input. |

#### Define the loss and optimization functions
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark: Select a loss function and optimizer for training the model. |  The loss and optimization functions should be appropriate for keypoint detection, which is a regression problem. |


#### Train the CNN

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:  Train your model.  |  Train your CNN after defining its loss and optimization functions. You are encouraged, but not required, to visualize the loss over time/epochs by printing it out occasionally and/or plotting the loss over time. Save your best trained model. |


#### Answer questions about model architecture

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:  All questions about model, training, and loss choices are answered.  | After training, all 3 questions in notebook 2 about model architecture, choice of loss function, and choice of batch_size and epoch parameters are answered. |


#### Visualize one or more learned feature maps

| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:  Apply a learned convolutional kernel to an image and see its effects. |  Your CNN "learns" (updates the weights in its convolutional layers) to recognize features and this step requires that you extract at least one convolutional filter from the trained model, apply it to an image, and see what effect this filter has on the image. |


#### Answer question about feature visualization
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:  After visualizing a feature map, answer: what do you think it detects? | This answer should be informed by how the filtered image (from the step above) looks. |


### Notebook 3

#### Detect faces in a given image
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:  Use a haar cascade face detector to detect faces in a given image. | The submission successfully employs OpenCV's face detection to detect all faces in a selected image. |

#### Transform each detected face into an input Tensor
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:  Turn each detected image of a face into an appropriate input Tensor. | You should transform any face into a normalized, square, grayscale image and then a Tensor for your model to take in as input (similar to what the `data_transform` did in Notebook 2). |

#### Predict and display the keypoints
| Criteria       		|     Meets Specifications	        			            | 
|:---------------------|:---------------------------------------------------------:| 
| :white_check_mark:  Predict and display the keypoints on each detected face. | After face detection with a Haar cascade and face pre-processing, apply your trained model to each detected face, and display the predicted keypoints on each face in the image. |


## Recommendations
* Your home folder (including subfolders) must be less than 2GB (/home/workspace)
* Your home folder (including subfolders) must be less than 25 megabytes to submit as a project.


## Bonus
* :white_check_mark: Initialize the weights of your CNN by sampling a normal distribution or by performing Xavier initialization so that a particular input signal does not get too big or too small as the network trains.
* :heavy_exclamation_mark: In Notebook 4, create face filters that add sunglasses, mustaches, or any .png of your choice to a given face in the correct location.
* :heavy_exclamation_mark: Use the keypoints around a person's mouth to estimate the curvature of their mouth and create a smile recognition algorithm .
* :heavy_exclamation_mark: Use OpenCV's k-means clustering algorithm to extract the most common facial poses (left, middle, or right-facing, etc.).
* :heavy_exclamation_mark: Use the locations of keypoints on two faces to swap those faces.
* :heavy_exclamation_mark: Add a rotation transform to our list of transformations and use it to do data augmentation.
* :heavy_exclamation_mark: Evaluate the advantage of substituting the old haar files from original [git](https://github.com/opencv/opencv/tree/master/data/haarcascades)
