# Devnagari-Character-Recognition
## Description:
Use OpenCV and Convolution Neural Network to recognize the Handwritten Devnagari Word text image.

### Dataset:
Downlod Dataset from this [link](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset#:~:text=UCI%20Machine%20Learning%20Repository%3A%20Devanagari%20Handwritten%20Character%20Dataset%20Data%20Set&text=Abstract%3A%20This%20is%20an%20image,and%20testing%20set(15%25))
### Requirements:
Download and cd to the directory where [requirements.txt](https://github.com/vivekagarwal2349/Devnagari-Character-Recognition/blob/main/requirements.txt) is located.
```markdown
$ pip install -r requirements.txt
```
     
## Image classification

<p align="center">
 <img  width="400" height="400" src="https://github.com/vivekagarwal2349/Devnagari-Character-Recognition/blob/main/src/cnn_model.png">
 <p align="center">
 <i>CNN Model Workflow</i><br> 
</p>
Input grayscale image of size 28x28, to the model in a batch size. Passing it through layers of CNN (Convolutional and Pooling) with 'relu' activation, including 1 flatten layer and 2 dense layer.

<p align="center">
 <img  width="450" height="500" src="https://github.com/vivekagarwal2349/Devnagari-Character-Recognition/blob/main/src/summary.png">
 <p align="center">
 <i>Model Summary</i><br> 
</p>

## Handeling Testing_images

Here we use **OpenCV** library such that it can read multiple word in an image, irrespective of individual orientation.
<p align="center">
 <img  width="350" height="300" src="https://github.com/vivekagarwal2349/Devnagari-Character-Recognition/blob/main/src/eg.jpg">
 <p align="center">
</p>
<p align="center">
 <img  width="350" height="300" src="https://github.com/vivekagarwal2349/Devnagari-Character-Recognition/blob/main/src/eg_0.png">
 <p align="center">
</p>
