# X_AI

This is simple package for your easy AI projects.You don't need to code, You don't need to know AI. Just install the module and just write 2 lines of code then you are done.In the backend it uses [tensorflow](https://www.tensorflow.org/) and [OpenCV](https://github.com/opencv/opencv)

## Installation
You can  simply use pip to install the latest version of X_AI

`pip install X_AI`

### ImageClassification

<pre>
from X_AI import ImageClassification
Classify =ImageClassification(path)    
Classify._Train()
</pre>

Here the path is to image folder. The folder must be of the form class1 -> contains all images of class1 and class2->for 2nd class images.

### ImageCompression

ImageCompression uses machine learning techniques(Unsupervised learning and Principal component Analysis(PCA) ) for Compressing the size of image.
<pre>
from X_AI import ImageCompression
compress = ImageCompression(image)
</pre>
