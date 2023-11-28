# Lung-Cancer-Detection-Using-Image-Processing-and-Deep-Learning-Techniques
This project aims to create a model using deep learning that can detect lung cancer at an earlier stage
## Abstract
This project aims to create a model using deep learning that can detect lung cancer at an earlier stage. A Convolutional Neural Network architecture is used to analyse the medical images of the lungs to classify them as malignant or benign.   The dataset used in the study comprises CT scan images of lung nodules from the publicly accessible LIDC-IDRI dataset. Transfer learning is used with the previously taught VGG-16 architecture to train the CNN model. The accuracy, precision, recall, and F1 score are some of the standard metrics utilised to evaluate the suggested model's performance. The findings show that the proposed model is capable of high accuracy and outperforms the existing methods that are considered to be state-of-the-art. The model that has been proposed has the potential to be of assistance to radiologists in the process of the early detection of lung cancer and to enhance the results for patients.
## PROPOSED METHODOLOGY
![image](https://github.com/chirag1902/Lung-Cancer-Detection-Using-Image-Processing-and-Deep-Learning-Techniques/assets/71887495/8b30db95-9b30-45e8-a0cb-709b0c7b12a0)

 
Fig 1: Proposed Architecture for Lung Cancer Detection

The Proposed Architecture has been divided into 4 main parts: image acquisition, image preprocessing, model building, and performance analysis. The first step, image acquisition, involves capturing the raw image data using a camera or sensor. The next step, image preprocessing, involves applying various techniques such as image resizing, resizing to prepare the image for analysis. The third step, model building, involves developing a deep learning or transfer learning model that can learn from the preprocessed images and perform the desired task of lung cancer classification. Finally, the performance analysis step involves evaluating the performance of the model by measuring metrics such as accuracy, precision, recall, and F1-score.

### Dataset Collection:

The dataset utilised in the study is the "Lung and Colon Cancer Histopathological Images" dataset [16], which is accessible on Kaggle. It includes histopathological images of lung and colon cancer samples. Andrew A. Borkowski generated the dataset by compiling and labelling the images. The collection consists of 250,000 histopathological images classified into five classifications. The images have a 768 × 768 pixels resolution and are saved in JPEG format. Each class in the dataset consists of 5000 instances. The five classes are lung benign tissue, lung adenocarcinoma, lung squamous cell carcinoma, colon adenocarcinoma, and colon benign tissue. The images are identified with their respective cancer and tissue  types, grouped as benign or malignant. The images of lung cancer are classified according to the five classes.
The images were generated from an original sample of HIPAA-compliant and validated sources, which included 750 images of lung tissue with 250 benign lung tissue, 250 lung adenocarcinomas, and 250 lung squamous cell carcinomas, and 500 images of colon tissue with 250 benign colon tissue and 250 colon adenocarcinomas; these were augmented to 25,000 using an augmentor package.

### Image Processing:

Changing an image's size is a typical image processing technique known as image Resizing. Before feeding input images into a neural network in the context of deep learning, image resizing is frequently employed to standardize the size of the images. Resizing ensures that all input images have the same dimensions because the majority of neural networks have a fixed input size.
The cvtColor function in OpenCV is a method for converting images from one color space to another. It can be used to change the color format of an image, in this project it is used to convert BGR to RGB. It is an essential tool for image processing tasks that involve color space manipulation. It allows for the creation of images in various color spaces and enables the application of specific color transformations to images.
Reshaping is an important technique in image processing as it helps to transform the dimensionality of an image while preserving the information content. This technique can be used to transform images from one shape to another, depending on the specific needs of the image processing application. For example, reshaping can be used to convert an image from a one-dimensional array to a two-dimensional matrix, which is more suitable for convolutional neural networks (CNNs). Overall, reshaping is a versatile technique that can be applied in many different image processing applications to improve the performance and accuracy of the algorithms used.

### Model Building:
	
#### CNN

 ![image](https://github.com/chirag1902/Lung-Cancer-Detection-Using-Image-Processing-and-Deep-Learning-Techniques/assets/71887495/def9c845-2fa3-4b68-8daf-a2d952e1ee51)

Fig 2: CNN Architecture

In order to interpret and categories visual images, neural networks of a certain type called convolutional neural networks (CNN) are used. In image recognition tasks including object detection, segmentation, and classification, it is frequently utilized. In this experiment, the lung cancer detection model was trained using a  model as the basic model. The Convolutional Neural Network (CNN) used in the lung cancer detection project involves several layers. The first layer is the input layer, which accepts the input image data. The input layer is followed by a series of convolutional layers, each of which performs a convolution operation on the input data to produce a set of output features. 
The output of the convolutional layers is then sent into a pooling layer, which takes the maximum or average value of a region of the feature map to down sample the output features. The output features are flattened into a one-dimensional vector after the pooling layer and transmitted through a fully connected layer, which is in charge of understanding the more complex characteristics of the data. 
The fully connected layer's output is then run through a softmax function, which normalises it. The CNN is trained with stochastic gradient descent (SGD), a variation of backpropagation that optimises the network weights and reduces the error between the expected and actual output labels.
The train-test split is a technique used in image processing to divide the dataset into two subsets: one for training the model and another for evaluating its performance. The goal of this split is to avoid the model over fitting on the training data, which is when the model becomes overly specialized to the training data and struggles to generalize to new data. The model's performance can be assessed on fresh, untested data by dividing the data into two subsets, allowing the model to be trained on one subset and tested on the other. 

Additionally, the train-test split enables model selection and hyper parameter tuning, where several models and parameters may be contrasted based on how well they perform on the test set. Ultimately, the train-test split is an essential step in developing accurate and reliable models for image processing tasks.

#### Efficient B0

![image](https://github.com/chirag1902/Lung-Cancer-Detection-Using-Image-Processing-and-Deep-Learning-Techniques/assets/71887495/b30ee06b-42bd-4993-801b-7386f2394a40)

Fig 3: Efficient B0 Architecture

A deep neural network architecture called EfficientB0 is developed to improve accuracy while utilizing fewer parameters and processing resources. To lower the 
computational cost while maintaining high accuracy, it combines bottleneck layers, squeeze-and-excitation modules, and depth-wise separable convolutions. In order to fine-tune the model for lung cancer diagnosis in this research, EfficientB0 was employed as a pre-trained model for transfer learning.

EfficientNetB0's architecture is made up of seven convolutional layer blocks. A stem convolutional layer, the first building block, uses the input image to extract basic features. The remaining six blocks are composed of a downsampling layer, a succession of convolutional layers, and the stem layer. The squeeze-and-excitation (SE) blocks used by EfficientNetB0 also learn to amplify relevant feature maps while suppressing less significant ones. The final classification output is generated by a fully connected layer after that. In comparison to other cutting-edge models, EfficientNetB0 has less parameters overall yet still performs very accurate on a variety of image classification tasks.

#### ResNet50

Convolutional neural network ResNet50 was created to address the issue of vanishing gradients in deep neural networks. It comprises 50 layers and makes advantage of residual connections to hasten convergence and boost accuracy. In order to fine-tune the model for lung cancer diagnosis, ResNet50 was utilized as a pre-trained model for transfer learning.

In this Project, the input to the ResNet50 network is a 64 x 64 x 3 image, and the first layer is a convolutional layer followed by a max-pooling layer. The next layers consist of several blocks of residual units. After the residual blocks, there is a global average pooling layer that averages the output of the last convolutional layer across all spatial locations. This is followed by a fully connected layer with 1000 output nodes. Overall, the ResNet50 architecture is very deep and complex, but its residual learning framework and skip connections between layers allow for more efficient training and better performance on image classification tasks.

#### InceptionV3

 ![image](https://github.com/chirag1902/Lung-Cancer-Detection-Using-Image-Processing-and-Deep-Learning-Techniques/assets/71887495/4ca68a1e-aad8-4667-83b1-c48025daec7a)

Fig 4: Inception V3 Architecture

A common architecture for image classification tasks is InceptionV3, which has been applied in a number of computer vision applications. By employing 1x1 convolutions to cut down on the number of channels in the feature maps before executing larger convolutions, InceptionV3 implements the concept of a "network in network" method. This increases effectiveness and lowers the number of parameters needed. Moreover, InceptionV3 makes use of the idea of "inception modules," which integrate several convolutional filter sizes in parallel to capture features at various scales. This enables the network to record both high-level information and fine-grained details in the images. The model can take advantage of the pre-trained weights to increase the precision of the lung cancer detection job by utilizing InceptionV3 as a feature extractor.

## RESULTS

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

Comparisons:

Models	Accuracy (in %)	Precision	Recall	F1 Score
CNN	95.85	0.959	0.959	0.959
Efficient B0	99.01	0.963	0.963	0.963
ResNet50	76.87	0.768	0.768	0.768
Inception V3	92.12	0.92	0.92	0.92
Table 1: Algorithms along with performance metrics

Models	Accuracy (in %)
	Class 1	Class 2	Class 3
CNN	93.14	99.64	95.05
Efficient B0	90.63	99.91	98.76
Inception V3	86.94	96.89	92.31
ResNet50	58.85	92.88	79.84


