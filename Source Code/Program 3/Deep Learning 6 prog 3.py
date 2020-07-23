from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
## load the libraries
from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from numpy import argmax, array_equal
import matplotlib.pyplot as plt
from keras.models import Model

from random import randint
import pandas as pd
import numpy as np
# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder

## input layer
input_layer = Input(shape=(784,))

## encoding architecture
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encode_layer1 = Dense(1500, activation='relu')(encoded)
encode_layer2 = Dense(1000, activation='relu')(encode_layer1)




## decoding architecture
decoded = Dense(784, activation='sigmoid')(encode_layer2)
decode_layer1 = Dense(1000, activation='relu')(decoded)
decode_layer2 = Dense(1500, activation='relu')(decode_layer1)

## output layer
output_layer  = Dense(784)(decode_layer2)

model = Model(input_layer, output_layer)



# this model maps an input to its encoded representation
model.compile(optimizer='adadelta', loss='binary_crossentropy')
from keras.datasets import mnist, fashion_mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

model.fit(x_train, x_train,
                epochs=5,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

preds = model.predict(x_test)


#introducing noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

model.fit(x_train_noisy, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_noisy))

preds_noisy = model.predict(x_test_noisy)

f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(x_test_noisy[i].reshape(28, 28))
plt.show()
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(preds_noisy[i].reshape(28, 28))
plt.show()
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history = model.fit(x_train,y_train, epochs=5, verbose=True, validation_data=(x_test,y_test), batch_size=256)