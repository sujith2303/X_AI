import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
import os

class objectdetection:
  def __init__(self,image_path=None,labels_path=None,same_directory=True,input_shape=(240,240,3)):
    '''
    same_directory is an boolean which tells whether the labels are in same directory
    of the class
    if it is false then set same_directory is false
    '''
    self.images_path = image_path
    self.labels_path = labels_path
    self.classes     = []
    self.images      = []
    self.labels      = []
    self.input_shape = input_shape
    self.same_directory = same_directory
    if not image_path:
      raise FileNotFoundError('Enter a valid path for images!!!!')
    if not labels_path and not same_directory:
      raise FileNotFoundError('Enter a valid path for labels!!!')
    self.load_data()
  
  def load_data(self):
    print('\n Loading data !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    if self.same_directory:
      self.classes = os.listdir(self.images_path)
      image_extensions = ['jpg','jpeg','png']
      label_extensions = ['txt']
      #loading the images
      for name in tqdm(self.classes):
        images_in_particular_class_path = os.path.join(self.images_path,name)
        filenames_of_image              = [fn for fn in os.listdir(images_in_particular_class_path)
              if any(fn.endswith(ext) for ext in image_extensions)]
        labels_filenames                = [fn for fn in os.listdir(images_in_particular_class_path)
              if any(fn.endswith(ext) for ext in label_extensions)]
        for image in filenames_of_image:
          path  = self.images_path + '/' + str(name) + '/'  + str(image)
          image = cv2.imread(path)
          image = cv2.resize(image,self.input_shape[:2])
          self.images.append(image)
        for label in labels_filenames:
          path = self.images_path + '/' + str(name) + '/' + str(label)
          with open(path,'r') as f:
            self.labels.append(f.read().strip())

      print('\n Successfully loaded all the data!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      
      
      
      
      
  def create_model(self):
    pass
  
  def TrainModel(self):
    pass
  
  
  
  
