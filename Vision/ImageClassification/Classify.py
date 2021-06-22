import numpy as np
import tensorflow as tf
import tqdm
import os
import matplotlib.pyplot as plt

__author__ = 'Sujith Anumala'

class ImageClassification:
  def __init__(self,ImagePath=None,LabelsPath=None,num_classes=None):
    self.ImagePath = ImagePath
    self.LabelsPath= LabelsPath
    self.classes   = classes
    self.images    = None
    self.labels    = None
    self.history   = None
    if not ImagePath:
      raise FileNotFoundError('Enter a valid path')
    if not LabelsPath:
      raise FileNotFoundError('Enter a valid path')
    if not num_classes:
      raise ValueError('Enter a Non-zero number')
  
  def Train(self,epochs=10,input_shape=(320,320,3),batch_size = 128):
    
    model =tf.keras.applications.vgg16.VGG16(
    include_top=False, weights='imagenet')
    
    x_input = tf.keras.layers.Input(shape = input_shape)
    x = model(x_input,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(100,activation='relu')(x_input)
    output=tf.keras.layers.Dense(self.classes,activation='Softmax')(x)
    
    model = tf.keras.Model(x_input,output,name='Classification Model')
    
    model.compile(loss='mse',optimizer='adam',metrics=['Accuracy'])
    
    history = model.fit(x=trainx,y=trainy,epochs,batch_size=batch_size)
    
    return model
  
  def plot_results(self):
      print('Plotting accuracy.................')
      plt.plot(history.history['accuracy'])
      plt.plot(history.history['val_accuracy'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()
      print('Plotting loss.................')
      plt.plot(history.history['loss'])
      plt.plot(history.history['val_loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train', 'test'], loc='upper left')
      plt.show()

      
    
