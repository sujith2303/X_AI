import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
import os

class objectdetection:
  def __init__(self,image_path=None,labels_path=None):
    self.images_path = image_path
    self.labels_path = labels_path
    if not image_path:
      raise FileNotFoundError('Enter a valid path for images!!!!')
    if not labels_path:
      raise FileNotFoundError('Enter a valid path for labels!!!')
    
  
  def load_data(self):
    pass
  
  def create_model(self):
    pass
  
  def TrainModel(self):
    pass
  
  
  
  
