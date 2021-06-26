import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2

__author__ = 'Sujith Anumala'


class ImageClassification:
    def __init__(self, ImagePath=None,input_shape=(320, 320, 3)):
        self.ImagePath = ImagePath
        self.classes = None
        self.images = []
        self.labels = []
        self.history = None
        self.filename = []
        self.input_shape = input_shape
        self.model = None
        self.prediction=None
        if not ImagePath:
            raise FileNotFoundError('Enter a valid path for images')

        self.get_train_data()
        self.num_classes = len(self.classes)

        #print(self.labels)


    def get_train_data(self):
        print('Loading images..............')
        filenames = os.listdir(self.ImagePath)
        self.classes = filenames #since filenames are classes i.e, we are storing
                                # the images under the class name folder 
        input_shape = self.input_shape[:2]
        for filename in tqdm(filenames):
            files = os.listdir(os.path.join(self.ImagePath,filename))
            for f in files:
                file = filename + '/' + f
                img = cv2.imread(os.path.join(self.ImagePath, file))
                try:
                    img = cv2.resize(img, input_shape)
                    img = np.array(img) 
                    self.images.append(img)
                    self.labels.append(self.classes.index(filename))
                    tf.keras.backend.clear_session()
                except:
                    pass
        self.images = np.array(self.images)
        self.labels = tf.keras.utils.to_categorical(self.labels)
        print('Successfully Loaded all the images.....')
        tf.keras.backend.clear_session()


    def _Train(self, epochs=10, batch_size=128):
        print()
        print()
        print()
        print('Training Your Model')
        input_shape = self.input_shape
        base_model = tf.keras.applications.vgg16.VGG16(
            include_top=False, input_shape=self.input_shape,weights='imagenet')
        base_model.trainable = False
        x_input = tf.keras.Input(shape=input_shape)
        x = base_model(x_input, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(100, activation='relu')(x)
        output = tf.keras.layers.Dense(self.num_classes, activation='Softmax')(x)

        model = tf.keras.Model(x_input, output, name='custom_model')
        if self.num_classes > 2:
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['Accuracy'])
        elif self.num_classes == 2:
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam', metrics=['Accuracy'])
        else:
            raise ValueError('Enter a value for num_classes greater than or equals to 2')

        self.history = model.fit(x=self.images,\
             y=self.labels,epochs= epochs,validation_split=0.1,\
              batch_size=batch_size)
        self.model = model
        print('Trained your Model Successfully!!!!!!!!!!!!!!!!!!!!!!!')
        print()
        print()
        print()
        # return model



    def plot_results(self):
        history = self.history
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


    def summary(self):
        print('Showing the summary of your model')
        self.model.summary()


    def save_model(self, path):
        print(f'Saving your model at {path} Location')
        path = str(path) + '/' + 'model.h5'
        self.model.save(path)


    def save_weights(self, path, save_format='.h5'):
        print(f'Saving weights at {path} location')
        if save_format == '.h5':
            path = str(path) + '/' + 'weights.h5'
            self.model.save_weights(path)
        elif save_format == '.weights':
            path = str(path) + '/' + 'weights.weights'
            self.model.save_weights(path)
        else:
            path = str(path) + '/' + 'weights' + str(save_format)
            self.model.save_weights(path)
        print(f'Successfully saved your weights at {path} location')

    def _predict(self,image_path=None):
        if not ImagePath:
            print('Please Provide a valid path')
            raise FileNotFoundError
        img = cv2.imread(image_path)
        img = cv2.resize(img,self.input_shape[:2])
        img = img/255
        img = np.expand_dims(img,axis=0)
        self.prediction=self.classes[np.argmax(self.model.predict(img))]
        print(f'Predicted a f{prediction}')
        print('type obj.prediction to access your predicted class!')


if __name__=='__main__':
    obj = ImageClassification(ImagePath='D:/ONEDRIVE/Desktop/images/images')
    obj._Train(epochs = 1)
    print(obj.summary())
    obj.plot_results()
    obj.plot_results()
