import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten,Conv2D,MaxPooling2D,Dropout,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array , load_img

from google.colab import drive
drive.mount('/content/drive')

base_path = '/content/drive/MyDrive/dataset'
train_path = '/content/drive/MyDrive/dataset/train'
test_path = '/content/drive/MyDrive/dataset/test'

image_gen = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,height_shift_range=0.1,rescale=1/255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

IMG_SHAPE = (128,128,3)

base_model = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
# Freezing the Base Model
base_model.trainable = False
#Define our Custom Head
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
prediction_layer = tf.keras.layers.Dense(1,activation='sigmoid')(global_average_layer)
# Define the Model
model = tf.keras.models.Model(inputs=base_model.input,outputs=prediction_layer)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

batch_size = 16
train_image_gen = image_gen.flow_from_directory(train_path,target_size=(224,224),batch_size=batch_size,class_mode='binary')

test_image_gen = image_gen.flow_from_directory(test_path,target_size=(128,128),batch_size=batch_size,class_mode='binary')

train_image_gen.class_indices

result = model.fit_generator(train_image_gen,epochs=100,steps_per_epoch=10,validation_data=test_image_gen,validation_steps=12)

model.save('BvsNo_10epoch_85_acc_model.h5')
model.evaluate(test_image_gen)

def pred_clean(pred):
  if pred > 0.6:
    return 1
  else:
    return 0
def pred_class(pred):
  res = pred_clean(pred)
  labels = ['Bulling','No Bulling']
  return labels[res]
 
img_file = '/content/drive/MyDrive/dataset/test/Bulling/images15.jpg'
image = load_img(img_file,target_size=(128,128))
plt.imshow(image)
image = img_to_array(image)
image = np.expand_dims(image,axis=0)
image = image / 255

img_file = '/content/drive/MyDrive/dataset/test/NoBulling/Friends-Hugging.png'
image = load_img(img_file,target_size=(128,128))
plt.imshow(image)
image = img_to_array(image)
image = np.expand_dims(image,axis=0)
image = image / 255

res = model.predict(image)
print(res)
pred = pred_class(res)
print(pred)
pred
  