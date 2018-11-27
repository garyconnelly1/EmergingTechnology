import gzip
import numpy as np
import tensorflow as tf
import numpy as np
import keras as kr
import sklearn.preprocessing as pre


import gzip

print("========================= IMAGES FILE IMPORTED ======================")
with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f: ### Opens the train images file.
    file_content = f.read() ### Loads the bits from the file into the file_content variable.

type(file_content)
print(type(file_content)) ### To ensure the file is in bytes.
print(file_content[0:4]) ### Print the first byte.
print(int.from_bytes(file_content[0:4], byteorder='big')) ### Convert the first byte to 'int'.
print(int.from_bytes(file_content[8:12], byteorder='big'))

### Read in the labels

with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f: ### Use gzip to open the labels file.
    labels = f.read() ### Read the bits from the file into the 'labels' variable.

print("========================= LABELS FILE IMPORTED ======================")
myInt = int.from_bytes(labels[4:8], byteorder="big") ### Reading the second byte which should contain the number of items in the file.

print(myInt) ### Output the number.

myInt = int.from_bytes(labels[8:9], byteorder="big") ### The label for number '5'.
print(myInt) ### Output the number.

print("======================== KERAS IMPORTED ===================") 

model = kr.models.Sequential() ### Start a neural network, building it by layers.
model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784)) ### Add a hidden layer with 1000 neurons and an input layer with 784.
model.add(kr.layers.Dense(units=400, activation='relu')) ### Using 'relu' activation function.

model.add(kr.layers.Dense(units=10, activation='softmax')) # Add a 10 neuron output layer.

print("Model Created!!") ### Test output.



