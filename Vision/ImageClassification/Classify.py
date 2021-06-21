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
  
  def Train(self):
    pass
    
    

      
    
