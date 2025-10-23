import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import cifar10
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
class Image:
    def __init__(self):
        (self.X_train,self.y_train),(self.X_test,self.y_test)=cifar10.load_data()
        self.X_train=self.X_train/255.0
        self.X_test=self.X_test/255.0
        self.y_train=to_categorical(self.y_train,10)
        self.y_test=to_categorical(self.y_test,10)
        self.class_names=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']
        self.history=None
        self.model=None

    def build_model(self):
        model=Sequential([
            layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
            layers.MaxPooling2D((2,2)),layers.Conv2D(64,(3,3),activation='relu'),
            layers.MaxPooling2D((2,2)),layers.Conv2D(128,(3,3),activation='relu'),
            layers.Dropout(0.4),
            layers.Flatten(),
            layers.Dense(128,activation='relu'),
            layers.Dense(10,activation='softmax')
        ])

        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model=model
        
    def augmenting_image(self):
        data=ImageDataGenerator(rotation_range=15,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
        data.fit(self.X_train)
        self.history=self.model.fit(data.flow(self.X_train,self.y_train,batch_size=64,),epochs=10,validation_data=(self.X_test,self.y_test),verbose=True)
        self.model.save('model_cifar10_v2.h5')

    ''' def evaluate(y_test,model):
        model.evaluate'''

    def plot(self):
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(self.history.history['accuracy'],label='Train Accuracy')
        plt.plot(self.history.history['val_loss'],label='Val accuracy')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.plot(self.history.history['loss'],label='loss Accuracy')
        plt.plot(self.history.history['val loss'],label='Val accuracy')
        plt.legend()
        plt.show()

if __name__=="__main__":
    i=Image()
    i.build_model()
    i.augmenting_image()
    i.plot()
