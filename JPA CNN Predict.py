import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam
import cv2

#load model
# I have commented this out as I still need to store this from JPA Deep Learning.py module
#model_architecture = 'stage-1_architecture.json'
#model_weights = 'stage-1_weights.h5'
model_architecture = 'cifar10_architecture.json'
model_weights = 'cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)

file = cv2.imread('cat.jpg')
file = cv2.resize(file, (32, 32))
#file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
#file = file.reshape(28, 28)

print(model.predict(np.expand_dims(file, axis = 0)))
