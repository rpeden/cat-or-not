from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from PIL import Image
from random import shuffle, choice
import numpy as np
import os

IMAGE_SIZE = 256
IMAGE_DIRECTORY = './data/training_set'

def label_img(name):
    if name == 'cats': return np.array([1, 0])
    elif name == 'notcats' : return np.array([0, 1])


def load_data():
  print("Loading images...")
  train_data = []
  directories = next(os.walk(IMAGE_DIRECTORY))[1]

  for dirname in directories:
    print("Loading {0}".format(dirname))
    file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, dirname)))[2]
    for i in range(200):
      image_name = choice(file_names)
      image_path = os.path.join(IMAGE_DIRECTORY, dirname, image_name)
      if image_name != ".DS_Store":
        label = label_img(dirname)
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        train_data.append([np.array(img), label])
  
  return train_data


def load_training_data():
  train_data = []
  directories = next(os.walk(IMAGE_DIRECTORY))

  for img in os.listdir(IMAGE_DIRECTORY):
      label = label_img(img)
      path = os.path.join(IMAGE_DIRECTORY, img)
      if "DS_Store" not in path:
          img = Image.open(path)
          img = img.convert('L')
          img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
          train_data.append([np.array(img), label])
          
  shuffle(train_data)
  return train_data

training_data = load_data()
training_images = np.array([i[0] for i in training_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
training_labels = np.array([i[1] for i in training_data])

print('loading model')
model = load_model("model.h5")
print('training model')
model.fit(training_images, training_labels, batch_size=50, epochs=5, verbose=1)
model.save("model.h5")