# Cotton Plant Disease Classification Web Application :
This repository is about an end to end implementation of deep learning cotton plant disease classification web application using flask. 

## Dataset
The dataset has a total size of 1 GB. The dataset contains approximately 14,000 train images and 14,000 test images, organized into two folders. Each folder is divided into two classes: Disease Affected and Pest Affected. Each image is in JPG format with a size of 256x256 pixels.

<img src="https://github.com/ShreeHarinesh1494/eveningdemo/blob/main/pdds.jpg" width=50% height=50%>

## Data Set Preparation :
### Model Information:
Base Model: DenseNet121 with pre-trained weights from ImageNet.
Input Shape: (256, 256, 3)
Output: 2 classes Pest Affected and Disease Affected


## Dataset Preparation Steps:
### Dataset Structure:
Organize your dataset into directories for each category. The dataset should be divided into:

Training Set
Test Set

Each set should contain the following categories:

Pest Affected
Disease Affected

### Image Resizing:
All images should be resized to (256, 256, 3) to match the model’s input shape.

### Data Augmentation:
Apply data augmentation techniques to the training dataset to improve model generalization. Common techniques include rescaling, rotating, flipping, and zooming.

### Data Splitting:
Split the dataset into training, validation, and test sets. A common split is 80% for training, 10% for validation, and 10% for testing.

## DenseNet Model
Pretrained DenseNet121 model on ImageNet dataset is used. With the help of transfer learning, the last 8 layers of the model are tuned to solve the problem. The model is trained for 20 epoches and the accuracy is 95% on test data. 

<img src="https://i.imgur.com/O8ntGzS.png">

## Training Accuracy and Loss
<img src="https://github.com/ShreeHarinesh1494/eveningdemo/blob/main/pdmlma.jpg">


## Demo
<img src="https://github.com/ShreeHarinesh1494/eveningdemo/blob/main/pddemo.jpg"  width=70% height=70%>

## Usage
• For model implementation and training, run densenet121cottondisease.ipynb.
• You can also directly download pest_model.h5 without running the notebook.
• To run Flask app, run app.py.
• Make sure that you did not change any folder name in this repo.


## Cmds to run file
- Installing dependencies
```
pip install -r requirements.txt
```

- Model Training  
```
pestordisease.ipynb
```

- Inference

Model weight available at - pest_model.h5 

To run Flask, use:
```
python app.py
```



