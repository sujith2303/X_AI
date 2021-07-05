import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import cv2

class ArtGeneration:
  def __init__(self,image_path=None,style_image_path=None,img_shape=(100,100,3)):
    self.img_nrows=img_shape[0]
    self.img_ncols=img_shape[1]
    if not image_path:
      raise FileNotFoundError('Please Enter a valid path for image_path!')
    if not style_image_path:
      raise FileNotFoundError('Please Enter a valid path for style_image!')
    baseimage = cv2.imread(image_path)
    baseimage = cv2.resize(baseimage,img_shape[:2])
    styleimage =cv2.imread(style_image_path)
    styleimage =cv2.resize(styleimage,img_shape[:2])
    print('Base image')
    cv2.imshow(baseimage)
    print('Style image')
    cv2.imshow(styleimage)
  
  def preprocess_image(self,image_path):
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(self.img_nrows, self.img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


  def deprocess_image(self,x):
      # Util function to convert a tensor into a valid image
      x = x.reshape((img_nrows, img_ncols, 3))
      # Remove zero-center by mean pixel
      x[:, :, 0] += 103.939
      x[:, :, 1] += 116.779
      x[:, :, 2] += 123.68
      # 'BGR'->'RGB'
      x = x[:, :, ::-1]
      x = np.clip(x, 0, 255).astype("uint8")
      return x

  def gram_matrix(self,tensors):
    return tf.matmul(tensors,tf.transpose(tensors))
  
  def content_cost(self,base,combined):
    m,n_H, n_W, n_C = combined.shape
    return (.25 / float(int(n_H * n_W * n_C))) *tf.reduce_sum(tf.square(combination - base))
  
  def style_cost(self,style,combined):
    m, n_H, n_W, n_C = combined.get_shape().as_list()
    S=self.gram_matrix(style)
    C=self.gram_matrix(combined)
    factor = (.5 / (n_H * n_W * n_C)) ** 2
    return factor * tf.reduce_sum(np.power(S - C, 2))
  
  def total_cost(self,J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    
    
    
    
    
    
  
  def content_cost(self,a_C,a_G):
    m, n_H, n_W, n_C = a_G.shape
    new_shape = [int(m), int(n_H * n_W), int(n_C)]
    a_C_unrolled = tf.reshape(a_C, new_shape)
    a_G_unrolled = tf.reshape(a_G, new_shape)
    J_content = (.25 / float(int(n_H * n_W * n_C))) * tf.reduce_sum(np.power(a_G_unrolled - a_C_unrolled, 2))
    return J_content
  
  def grammatrix(self,A):
    GA = tf.matmul(A, tf.transpose(A)) 
    return GA

  def style_cost(self,a_S,a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S, [n_H * n_W, n_C])
    a_S = tf.transpose(a_S)
    a_G = tf.reshape(a_G, [n_H * n_W, n_C])
    a_G = tf.transpose(a_G)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    factor = (.5 / (n_H * n_W * n_C)) ** 2
    J_style_layer = factor * tf.reduce_sum(np.power(GS - GG, 2))
    return J_style_layer
  
  def compute_style_cost(self,model,style_layers):
    J_style = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer
    return J_style
  
  def total_cost(self,J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta * J_style
    return J
  def model_nn(self,sess, input_image, num_iterations = 200):
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    for i in range(num_iterations):
        somthing = sess.run(train_step)
        generated_image = sess.run(model['input'])
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            save_image("output/" + str(i) + ".png", generated_image)
    save_image('output/generated_image.jpg', generated_image)
    return generated_image
