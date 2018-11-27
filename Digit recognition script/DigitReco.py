import gzip
import numpy as np
import tensorflow as tf
import numpy as np
import keras as kr
import sklearn.preprocessing as pre
import sys
import gzip

### Functions:

def readTrainImages():
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f: ### Opens the train images file.
        file_content = f.read() ### Loads the bits from the file into the file_content variable.
    return file_content

def readTrainLabels():
    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f: ### Use gzip to open the labels file.
        labels = f.read() ### Read the bits from the file into the 'labels' variable.
    return labels

def readTestImages():
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_img = f.read() ### Read in the 10000 test images
    return test_img

def readTestLabels():
    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read() ### Read in the corresponding 10000 test labels
    return test_lbl

def createModelRelu():
    model = kr.models.Sequential() ### Start a neural network, building it by layers.
    model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784)) ### Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=400, activation='relu')) ### Using 'relu' activation function.

    model.add(kr.layers.Dense(units=10, activation='softmax')) ### Add a 10 neuron output layer.
    return model

def createModelSigmoid():
    model = kr.models.Sequential() ### Start a neural network, building it by layers.
    model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784)) ### Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=400, activation='sigmoid')) ### Using 'relu' activation function.

    model.add(kr.layers.Dense(units=10, activation='softmax')) ### Add a 10 neuron output layer.
    return model

### tanh
def createModelTanh():
    model = kr.models.Sequential() ### Start a neural network, building it by layers.
    model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784)) ### Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=400, activation='tanh')) ### Using 'relu' activation function.

    model.add(kr.layers.Dense(units=10, activation='softmax')) ### Add a 10 neuron output layer.
    return model


### End functions.    

print("PRESS 1: ------------------------> Relu")
print("PRESS 2: ------------------------> Sigmoid")
print("PRESS 3: ------------------------> Tanh")
activationFunction = int(input("Select the activation function you wish to use: "))
print("PRESS 1: ------------------------> Adam")
print("PRESS 2: ------------------------> Stochastic gradient descent")
print("PRESS 3: ------------------------> RMSProp")
selectedOptimizer = int(input("Select the optimizer you wish to use: "))

print (activationFunction)
print(selectedOptimizer)

file_content = readTrainImages() ### Call readTrainImages function.
print("========================= IMAGES FILE IMPORTED ======================")
type(file_content)
print(type(file_content)) ### To ensure the file is in bytes.
print(file_content[0:4]) ### Print the first byte.
print(int.from_bytes(file_content[0:4], byteorder='big')) ### Convert the first byte to 'int'.
print(int.from_bytes(file_content[8:12], byteorder='big'))

### Read in the labels

labels = readTrainLabels() ### Call readTrainLabels function


print("========================= LABELS FILE IMPORTED ======================")
myInt = int.from_bytes(labels[4:8], byteorder="big") ### Reading the second byte which should contain the number of items in the file.

print(myInt) ### Output the number.

myInt = int.from_bytes(labels[8:9], byteorder="big") ### The label for number '5'.
print(myInt) ### Output the number.

print("======================== KERAS IMPORTED ===================") 


### Create a model using the activation function the user selected.

if (activationFunction == 1):
    model = createModelRelu() ### Create the model using 'relu' activation function.
elif (activationFunction == 2):
    model = createModelSigmoid() ### Create the model using 'sigmoid' activation function.
elif (activationFunction == 3):
    model = createModelTanh() ### Create the model using 'tanh' activation function.


print("Model Created!!") ### Test output.
optimizer = 'adam'
optimizerSQD = 'sgd'
optimizerRMSprop = 'rmsprop'
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) ### Build the graph.

file_content = ~np.array(list(file_content[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0 ### Take every bit after the 16th bit of the file_content variable, resize it into 60000 28*28 arrays and divide by 255 to map it to a color code.
labels =  np.array(list(labels[ 8:])).astype(np.uint8) ### Take every bit after the 8th bit of the labels array.

inputs = file_content.reshape(60000, 784) ### Reshape the file_ccontent because the neural network has 784 input nodes.

encoder = pre.LabelBinarizer() ### So that we can view a number between 1-10 as a series of 0s and 1s.
encoder.fit(labels) ### Encode the labels variable.
outputs = encoder.transform(labels) ### Transform the labels using the encoder.

print(labels[0], outputs[0]) ### Test output to see if the encoder is working.
### CONVERT EPOCH BACK TO 4
model.fit(inputs, outputs, epochs=1, batch_size=100) ### Fit the model with 100 elements at a time(Faster precessing) and do this 4 times(Better training).

print("4 Epochs done") ### Debug output to show that is has finished.

test_img = readTestImages() ### Read the 10k test images file.

print("========================= TEST IMAGES FILE IMPORTED ======================")

test_lbl = readTestLabels() ### Read the 10k test labels file.

print("========================= TEST LABELS FILE IMPORTED ======================")

test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0 ### This time, we reshape it to 10000 784 element arrays.
test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)

print((encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()) ### Print out the amount of digits it correctly predicts by comparing it to the corresponding element in the labels file


for i in range(10):
    result = (encoder.inverse_transform(model.predict(test_img[i:i+1]))) ### Output the first 10 elements of the return array and labels array so the user can visualize it better.
    print(result)
    print(test_lbl[i])


