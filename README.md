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
