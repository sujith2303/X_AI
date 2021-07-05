# X_AI

This is simple package for your easy AI projects.
Training our own models is hard and you need to know tougher frameworks like [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/). But X_AI made it easy. You don't need to code to train your own projects be it simple cat,dog classification or be it complex object detection; be it a 100 layered deep network or just one layered network. Size varies but your code length is constant. All you need is just 3-4 lines of python code.
You don't need to code, You don't need to know AI. Just install the module and just write 2 lines of code then you are done.

## Installation
You can  simply use pip to install the latest version of X_AI

`pip install X_AI`

## ImageClassification:-

<p align="center">
  <img width="640" height="360" src="https://github.com/sujith2303/X_AI/blob/main/images/Cats-Dogs-Classification-deep-learning.gif">
</p>
<pre>
from X_AI import ImageClassification
Classify =ImageClassification(path)    
Classify._Train()
</pre>
To test:-
<pre>
Classify.predict(img)
</pre>

Here the path is to image folder. The folder must be of the form class1 -> contains all images of class1 and class2->for 2nd class images.


## ImageCompression:-


<p align="center">
  <img width="640" height="360" src="https://github.com/sujith2303/X_AI/blob/main/images/compression.jpg">
</p>

ImageCompression uses machine learning techniques(Unsupervised learning and Principal component Analysis(PCA) ) for Compressing the size of image.
<pre>
from X_AI import ImageCompression
import cv2
compress = ImageCompression(image)
cv2.imshow(compress)
</pre>


## Neural Style Transfer:-

<hr>

<p align="center">
  <img width="640" height="360" src="https://github.com/sujith2303/X_AI/blob/main/images/NST.png">
</p>

<pre>
from X_AI import NeuralStyle
import cv2
nst = NeuralStyle(image,styleimage)
cv2.imshow(nst.img)
</pre>
image is path to image
styleimage is path to style image

## Object Detection:-
Train your own custom object detection with just 2 lines of code.Performing object detection is difficult for python users to understand [darknet](https://pjreddie.com/darknet). X_AI made it easy. Write just 4 lines of python code and train your own model (be it detecting faces /hands /yourself/ anything).

<pre>
from X_AI import objectdetection 
obj = objectdetection._Train(img_path,labels_path)
</pre>
img_path --- path to images 
labels_path --- path to labels (including bounding boxes)


## Virtual game:-
Virtual game is a program where a cricket bat is tracked by our camera. We track the bat movement and corresponding command is sent to the controller. We estimate the pose (standing style) of the player , the bat's movement and give commands in the game.
