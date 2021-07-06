import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import cv2

class ArtGeneration:
  def __init__(self,image_path=None,style_image_path=None,img_shape=(100,100,3)):
    self.img_nrows=img_shape[0]
    self.img_ncols=img_shape[1]
    self.model =None
    self.iteration = 1000
    self.style_layer_names = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    self.content_layer_name = "block5_conv2"
    self.content_weight = 10
    self.style_weight = 40
    if not image_path:
      raise FileNotFoundError('Please Enter a valid path for image_path!')
    if not style_image_path:
      raise FileNotFoundError('Please Enter a valid path for style_image!')
    baseimage = cv2.imread(image_path)
    styleimage =cv2.imread(style_image_path)
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
    
  def _model(self):
    model = vgg19.VGG19(weights="imagenet", include_top=False)
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
    self.model =feature_extractor

  def compute_loss(self,base,style,combined):
    input_tensor = tf.concat([
      base,style,combined],axis=0)
    features = self.model(input_tensor)
    loss = tf.zeros(shape=())
    
    # Add content loss
    layer_features = features[self.content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + self.content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in self.style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (self.style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss
  
  @tf.function
  def compute_loss_and_grads(self,combination_image, base_image, style_reference_image):
      with tf.GradientTape() as tape:
          loss = self.compute_loss( base_image, style_reference_image,combination_image)
      grads = tape.gradient(loss, combination_image)
      return loss, grads  

  def main(self):
    optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

    base_image = self.preprocess_image(self.base_image_path)
    style_reference_image = self.preprocess_image(self.style_reference_image_path)
    combination_image = tf.Variable(self.preprocess_image(self.base_image_path))
    for i in range(1, self.iterations + 1):
        loss, grads = self.compute_loss_and_grads(
            combination_image, base_image, style_reference_image
        )
        optimizer.apply_gradients([(grads, combination_image)])
        if i % 100 == 0:
            print("Iteration %d: loss=%.2f" % (i, loss))
            img = deprocess_image(combination_image.numpy())
            fname = result_prefix + "_at_iteration_%d.png" % i
            keras.preprocessing.image.save_img(fname, img)
            
