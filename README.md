# X_AI

This is simple package for your easy AI projects.You don't need to code, You don't need to know AI. Just install the module and just write 2 lines of code then you are done.In the backend it uses [tensorflow](https://www.tensorflow.org/) and [OpenCV](https://github.com/opencv/opencv)

## Installation
You can  simply use pip to install the latest version of X_AI

`pip install X_AI`

### ImageClassification
<p align="center">
  <img width="640" height="360" src="https://data-flair.training/blogs/wp-content/uploads/sites/2/2020/05/Cats-Dogs-Classification-deep-learning.gif">
</p>
<pre>
from X_AI import ImageClassification
Classify =ImageClassification(path)    
Classify._Train()
</pre>

Here the path is to image folder. The folder must be of the form class1 -> contains all images of class1 and class2->for 2nd class images.

### ImageCompression
<p align="center">
  <img width="640" height="360" src="https://www.lifewire.com/thmb/dpgls7GZ2zGXbAl3U3z8L7hnHos=/768x0/filters:no_upscale():max_bytes(150000):strip_icc()/JPEG_compression_Example-ibrahim-id-5811001d3df78c2c73161f7c.jpg">
</p>

ImageCompression uses machine learning techniques(Unsupervised learning and Principal component Analysis(PCA) ) for Compressing the size of image.
<pre>
from X_AI import ImageCompression
compress = ImageCompression(image)
</pre>


### Neural Style Transfer

<hr>

<p align="center">
  <img width="640" height="360" src="https://miro.medium.com/max/2396/1*kOQOZxBDNw4lI757soTEyQ.png">
</p>

<pre>
from X_AI import NeuralStyle
import matplotlib.pyplot as plt
nst = NeuralStyle(image,styleimage)
plt.imshow(nst.img)
</pre>
image is path to image
styleimage is path to style image
