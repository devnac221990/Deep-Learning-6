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

history = model.fit(x_train, x_train,
                epochs=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

preds = model.predict(x_test)

f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(x_test[i].reshape(28, 28))
plt.show()
f, ax = plt.subplots(1,5)
f.set_size_inches(80, 40)
for i in range(5):
    ax[i].imshow(preds[i].reshape(28, 28))
plt.show()


results = model.evaluate(x_test, x_test)
print("_"*100)
print("Test Loss and Accuracy")
print("results ", results)
history_dict = history.history
history_dict.keys()

# VALIDATION LOSS curves

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'])
plt.show()
# VALIDATION ACCURACY curves



plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'])
plt.show()