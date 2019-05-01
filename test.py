from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from PIL import Image
from random import shuffle, choice
import numpy as np
import os

IMAGE_SIZE = 256
IMAGE_DIRECTORY = './data/test_set'

def label_img(name):
    if name == 'cats': return np.array([1, 0])
    elif name == 'notcats' : return np.array([0, 1])


def load_data():
  print("Loading images...")
  test_data = []
  directories = next(os.walk(IMAGE_DIRECTORY))[1]

  for dirname in directories:
    print("Loading {0}".format(dirname))
    file_names = next(os.walk(os.path.join(IMAGE_DIRECTORY, dirname)))[2]
    for i in range(len(file_names)):
      image_name = choice(file_names)
      image_path = os.path.join(IMAGE_DIRECTORY, dirname, image_name)
      if image_name != ".DS_Store":
        label = label_img(dirname)
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
        test_data.append([np.array(img), label])
  
  return test_data

test_data = load_data()
test_images = np.array([i[0] for i in test_data]).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
test_labels = np.array([i[1] for i in test_data])

print('Loading model...')
model = load_model("model.h5")

print('Testing model...')
loss, acc = model.evaluate(test_images, test_labels, verbose=1)

print("accuracy: {0}".format(acc * 100))