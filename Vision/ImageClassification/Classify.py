import numpy as np
import tensorflow as tf
import tqdm
import os

__author__ = 'Sujith Anumala'

class ImageClassification:
  def __init__(self,ImagePath=None,LabelsPath=None,num_classes=None):
    self.ImagePath = ImagePath
    self.LabelsPath= LabelsPath
    self.classes   = classes
    self.images    = None
    self.labels    = None
    if not ImagePath:
      raise FileNotFoundError('Enter a valid path')
    if not LabelsPath:
      raise FileNotFoundError('Enter a valid path')
    if not num_classes:
      raise ValueError('Enter a Non-zero number')
  
  def Train(self,epochs=10,input_shape=(320,320,3):
    
    model =tf.keras.applications.vgg16.VGG16(
    include_top=True, weights='imagenet',input_shape=input_shape, classes=self.classes,
    classifier_activation='softmax')
    
    model.compile(loss='mse',optimizer='adam',metrics=['Accuracy'])
    history = model.fit(x=trainx,y=trainy,epochs)
    return model
    
    

      
    
