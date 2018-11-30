import gzip
import numpy as np
import tensorflow as tf
import numpy as np
import keras as kr
import sklearn.preprocessing as pre
import sys
import gzip
import cv2
from PIL import Image
from keras.models import load_model
from keras.layers import Dropout

import SaveImages as save

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

def createModelTanh():
    model = kr.models.Sequential() ### Start a neural network, building it by layers.
    model.add(kr.layers.Dense(units=600, activation='linear', input_dim=784)) ### Add a hidden layer with 1000 neurons and an input layer with 784.
    model.add(kr.layers.Dense(units=400, activation='tanh')) ### Using 'relu' activation function.

    model.add(kr.layers.Dense(units=10, activation='softmax')) ### Add a 10 neuron output layer.
    return model

def predictAll(encoder, model, test_img, test_lbl):
    print((encoder.inverse_transform(model.predict(test_img)) == test_lbl).sum()) ### Print out the amount of digits it correctly predicts by comparing it to the corresponding element in the labels file
    print("^ Images predicted correctly out of 10,000")
    for i in range(10):
        result = (encoder.inverse_transform(model.predict(test_img[i:i+1]))) ### Output the first 10 elements of the return array and labels array so the user can visualize it better.
        print("PREDICTION: " + str(result))
        print("ACTUAL: " + str(test_lbl[i]))

def readImageAsBytes(imageName): ### Function to read in a png file from file system.
    b = cv2.imread(imageName + ".png") ### Read in the file. It will be in format [28(28*3)].

    if b is None:
        print("File does not exist or is not an image file. Try again") ### Error message to tell the user why it failed
        sys.exit(0)

    gray = cv2.cvtColor(b,cv2.COLOR_BGR2GRAY) ### Change the RGB values to gray scale values so the network can process them. File is now in format [28*28].
    img = np.array(gray).reshape(784,1)/255 ### Reshape the image to an array of shape [784*1].

    return img ### Return that array.

def predictFile(b, model, encoder): ### Function to predict the contents of a file passed from the file system.
     bArray = np.array(list(b)).reshape(1, 784).astype(np.uint8) ### Reshape the array to [1*784] so it fits the network(784 input neurons).
    
     result = (encoder.inverse_transform(model.predict(bArray))) ### Create a result.
     print("PREDICTION: " + str(result)) ### Output prediction
     print("================= SINGLE FILE PREDICTED ======================")

def passFile(model, encoder): ### Function to allow the user to pass a selected file network.
    fileName = input("Please enter the name of the file you wish to test.(Just the file name, no extensions)") ### Instruct the user how to preceed.
    theImageFile = readImageAsBytes(fileName) ### Turn the image into a gray scale array of the correct format.
    predictFile(theImageFile, model, encoder) ### Predict the result.
    
    
### End functions.

isRunning = "Y"
while isRunning == "Y":
    print("----------------------- WELCOME ------------------------")
    print("PRESS 1: ------------------------> Load one of the already trained models in memory.")
    print("PRESS 2: ------------------------> Configure and train your own model (Uses 2 epochs by default)")
    useDefault = int(input("Select how you want to interact with the program: "))

    file_content = readTrainImages() ### Call readTrainImages function.
    print("========================= IMAGES FILE IMPORTED ======================")
    type(file_content)

    ### Read in the labels

    labels = readTrainLabels() ### Call readTrainLabels function


    print("========================= LABELS FILE IMPORTED ======================")
    myInt = int.from_bytes(labels[4:8], byteorder="big") ### Reading the second byte which should contain the number of items in the file.

    myInt = int.from_bytes(labels[8:9], byteorder="big") ### The label for number '5'.
   
    print("Loading(Please wait a moment)...")
    

    file_content = ~np.array(list(file_content[16:])).reshape(60000, 28, 28).astype(np.uint8) / 255.0 ### Take every bit after the 16th bit of the file_content variable, resize it into 60000 28*28 arrays and divide by 255 to map it to a color code.
    labels =  np.array(list(labels[ 8:])).astype(np.uint8) ### Take every bit after the 8th bit of the labels array.

    inputs = file_content.reshape(60000, 784) ### Reshape the file_ccontent because the neural network has 784 input nodes.

    encoder = pre.LabelBinarizer() ### So that we can view a number between 1-10 as a series of 0s and 1s.
    encoder.fit(labels) ### Encode the labels variable.
    outputs = encoder.transform(labels) ### Transform the labels using the encoder.
    
    if (useDefault == 2):
        ### Allow the user to select the activation function they wish to use.
        print("======================================================")
        print("PRESS 1: ------------------------> Relu")
        print("PRESS 2: ------------------------> Sigmoid")
        print("PRESS 3: ------------------------> Tanh")
        activationFunction = int(input("Select the activation function you wish to use: "))

        ### Allow the user to select the optimizer they wish to use.
        print("======================================================")
        print("PRESS 1: ------------------------> Adam")
        print("PRESS 2: ------------------------> Stochastic gradient descent")
        print("PRESS 3: ------------------------> RMSProp")
        selectedOptimizer = int(input("Select the optimizer you wish to use: "))

        ### Create a model using the activation function the user selected.

        if (activationFunction == 1):
             model = createModelRelu() ### Create the model using 'relu' activation function.
        elif (activationFunction == 2):
            model = createModelSigmoid() ### Create the model using 'sigmoid' activation function.
        elif (activationFunction == 3):
            model = createModelTanh() ### Create the model using 'tanh' activation function.
        else:
            print("Not one of the options, please re run the program and try again.")
            sys.exit(0)


        print("Model Created!!") ### Test output.

        ### Use the optimizer the user selected.

        if (selectedOptimizer == 1):
            optimizer = 'adam'
        elif (selectedOptimizer == 2):
            optimizer = 'sgd'
        elif (selectedOptimizer == 3):
            optimizer = 'rmsprop'
        else:
            print("Not one of the options, please re run the program and try again.")
            sys.exit(0)

        print("OPTIMIZER: ", optimizer)
        

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) ### Build the graph.
        model.fit(inputs, outputs, epochs=2, batch_size=100) ### Fit the model with 100 elements at a time(Faster precessing) and do this 4 times(Better training).
       # model.save('my_model5Epochs')

    elif (useDefault == 1):
        print("---------------- WHAT MODEL WOULD YOU LIKE TO LOAD? -------------")
        print("PRESS 1: ------------------------> Model trained with 1 epoch")
        print("PRESS 2: ------------------------> Model trained with 2 epochs")
        print("PRESS 3: ------------------------> Model trained with 3 epochs")
        print("PRESS 4: ------------------------> Model trained with 4 epochs")
        print("PRESS 5: ------------------------> Model trained with 5 epochs")
        whatModelToUse = int(input("Select the trained model you would like to load from disk: "))

        ### Use the saved model the user selected.

        if (whatModelToUse == 1):
             model = load_model('my_model1Epochs')
        elif (whatModelToUse == 2):
            model = load_model('my_model2Epochs')
        elif (whatModelToUse == 3):
            model = load_model('my_model3Epochs')
        elif (whatModelToUse == 4):
            model = load_model('my_model4Epochs')
        elif (whatModelToUse == 5):
            model = load_model('my_model5Epochs')
        else:
            print("Not one of the options, please re run the program and try again.")
            sys.exit(0)
            
        print("model loaded from disk")


    test_img = readTestImages() ### Read the 10k test images file.

    print("========================= TEST IMAGES FILE IMPORTED ======================")

    test_lbl = readTestLabels() ### Read the 10k test labels file.

    print("========================= TEST LABELS FILE IMPORTED ======================")

    test_img = ~np.array(list(test_img[16:])).reshape(10000, 784).astype(np.uint8) / 255.0 ### This time, we reshape it to 10000 784 element arrays.
    test_lbl =  np.array(list(test_lbl[ 8:])).astype(np.uint8)


    print("\n\n ======================== SELECT THE TESTING METHOD ====================") ### Allow the user to select how to test the network.
    print("PRESS 1: -------------------------> Test all images, showing you the sum of correctly predicted images.(Along with outputting the predicted vs actual results for the first 10)")
    print("PRESS 2: -------------------------> Automatically install 20 random images so you can pass a selected file to the network and view the predicted result.")
    methodOfTesting = int(input("Select the method of testing you wish to use: "))
   
    if (methodOfTesting == 1):
         predictAll(encoder, model, test_img, test_lbl) ### Predict all the images and output the first 10.
    elif (methodOfTesting == 2):
        print("Saving the images...")
        save.saveImages()  ### Save the images to the devices file system.
        passFile(model, encoder) ### Allow the user to enter the file they wish to test.
    else:
        print("Not one of the options, please re run the program and try again.")
        sys.exit(0)
        
   
    isRunning = input("Try again?: [Y/N] ") ### Give the user the opportunity to re-run the program.

print("Program Exited!")

