import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2
from keras.layers import Dense,Flatten,add


__author__ = 'Sujith Anumala'


class ImageClassification:
    def __init__(self, ImagePath=None,input_shape=(320, 320, 3)):
        self.ImagePath = ImagePath
        self.classes = None
        self.images = []
        self.labels = []
        self.history = None
        self.input_shape = input_shape
        self.model = None
        self.prediction=None
        self.valimages=None
        self.vallabels=None
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-5)
        if not ImagePath:
            raise FileNotFoundError('Enter a valid path for images')

        print('Loading images.................!!!!!!!!!!!!!!')
        self.get_train_data()
        self.num_classes = len(self.classes)


    def get_train_data(self):
        filenames = os.listdir(self.ImagePath)
        self.classes = filenames #since filenames are classes i.e, we are storing
                                # the images under the class name folder 
        input_shape = self.input_shape[:2]
        for filename in filenames:
            files = os.listdir(os.path.join(self.ImagePath,filename))
            for f in tqdm(files):
                filed = filename + '/' + f
                img = cv2.imread(os.path.join(self.ImagePath, filed))
                try:
                    img = cv2.resize(img, input_shape)
                    self.images.append(img)
                    self.labels.append(self.classes.index(filename))
                    tf.keras.backend.clear_session()
                except:
                    pass
        self.images = np.array(self.images)
        self.labels = tf.keras.utils.to_categorical(self.labels)
        print('Successfully Loaded all the images.....')
        tf.keras.backend.clear_session()
        np.random.seed(1)
        np.random.shuffle(self.images)
        np.random.shuffle(self.labels)
        n =len(self.labels)
        if  n> 550:
            self.images = self.images[:500]
            self.labels = self.labels[:500]
            self.valimages=self.images[500:550]
            self.vallabels =self.labels[500:550]
        else:
            self.images = self.images[:0.9*n]
            self.labels = self.labels[:0.9*n]
            self.valimages=self.images[0.9*n:]
            self.vallabels =self.labels[0.9*n:]


    def _Train(self, epochs=10, batch_size=128,model=None,_save=False):
        print()
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,   #Scale the image between 0 and 1
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
        train_generator = train_datagen.flow(self.images, self.labels,batch_size=batch_size)
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
        val_generator = val_datagen.flow(self.valimages, self.vallabels, batch_size=batch_size)
        
        print('Training Your Model')
        input_shape = self.input_shape
        if not model:
            base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet',\
                                                               include_top=False, input_shape=self.input_shape)
            base_model.trainable = False
            model=tf.keras.models.Sequential()
            model.add(base_model)
            model.add(Flatten())
            model.add(Dense(100,activation='relu'))
            model.add(Dense(self.num_classes,activation='softmax'))
        if self.num_classes > 2:
            model.compile(loss='categorical_crossentropy', optimizer=self.optimizer,\
                          metrics=['acc'])
        elif self.num_classes == 2:
            model.compile(loss='binary_crossentropy', optimizer=self.optimizer,\
                          metrics=['acc'])
        else:
            raise ValueError('Enter a value for num_classes greater than or equals to 2')
        if _save:
          filepath = 'model.epoch{epoch:02d}-loss{val_loss:.2f}.h5'
          checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
                             monitor='val_loss',
                             verbose=1, 
                             save_best_only=True,
                             mode='min')
          callbacks = [checkpoint]
          self.history = model.fit( train_generator,
                                    steps_per_epoch=len(self.labels) // batch_size,
                                    epochs=epochs,
                                   validation_data=val_generator,
                                validation_steps=nval // batch_size,
                                    callbacks=callbacks)
        else:
          self.history = model.fit( train_generator,
                                    steps_per_epoch=len(self.labels) // batch_size,
                                   validation_data=val_generator,
                              validation_steps=nval // batch_size,
                                    epochs=epochs
                                    )
        self.model = model
        print('Trained your Model Successfully!!!!!!!!!!!!!!!!!!!!!!!')
        print()
        print()
        print()



    def plot_results(self):
        if not self.history:
            raise Exception('Unable to plot! Make Sure You Trained your Model!')
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
        if not self.model:
            raise Exception('Unable to show summary! Make Sure You created a Model!')
        print('Showing the summary of your model')
        self.model.summary()


    def save_model(self, path):
        if not self.model:
            raise Exception('Make Sure You Trained your Model!')
        print(f'Saving your model at {path} Location')
        path = str(path) + '/' + 'model.h5'
        self.model.save(path)


    def save_weights(self, path, save_format='.h5'):
        if not self.model:
            raise Exception('Make Sure You Trained your Model!')
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
        if not self.history:
            raise Exception('Make Sure You Trained your Model!')
        if not image_path:
            print('Please Provide a valid path')
            raise FileNotFoundError
        img = cv2.imread(image_path)
        img = cv2.resize(img,self.input_shape[:2])
        img = img/255
        img = np.expand_dims(img,axis=0)
        self.prediction=self.classes[np.argmax(self.model.predict(img))]
        print(f"Predicted a {self.prediction}")
        print('type obj.prediction to access your predicted class!')


if __name__=='__main__':
    obj = ImageClassification(ImagePath='D:/ONEDRIVE/Desktop/images/images')
    obj._Train(epochs = 1)
    print(obj.summary())
    obj.plot_results()
    obj.plot_results()

