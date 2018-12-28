# This is an example of Multi Layer Precptron (MLP). It is called Multilayered because if you look at the example you will find that
# there are multiple activation layers (sigmoid), multiple dropout layers and dense layers.

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
#model is sequential because we can add as many layers one at a time next to each other.
# The random seed sets the same random seed every time so the output is consistent
np.random.seed(1671)
#No of Iterations this model has to undergo learning
NB_EPOCH = 30
#batch_size improves the qulity of your learning
BATCH_SIZE = 512
# meaning you will see a ====== line progress
VERBOSE = 1
#classes determine how many output functions you can have. Here we are predicting number of digits from 0-9 and hence 10 output classes.
NB_CLASSES = 10
#The options are SGD, Adam, Ada..,RMSProp etc
OPTIMIZER = Adam()
# The number of hidden layers in the network
N_HIDDEN = 128
# This means 10% at random from the Input Data Set isused for validating the model
VALIDATION_SPLIT=0.1
# This means 20% is dropped of from the input at each Hidden layer
DROPOUT = 0.2

#2 tuples:
#x_train, x_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
#y_train, y_test: uint8 array of digit labels (integers in range 0-9) with shape (num_samples,).
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#in MNIST the input data are 28x28 images
RESHAPED = 784

X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
'''
#simple neural network
model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
model.summary()
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])
#ReLu & Hidden layer
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
model.summary()
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])
#Regularization
'''
model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
history=model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH,verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
model.summary()
score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("Test score:", score[0])
print('Test accuracy:', score[1])


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()