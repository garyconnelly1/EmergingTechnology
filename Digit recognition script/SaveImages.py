import gzip ### This package allows us to unzip the file for reading.
import numpy as np ### Importing the numpy package.
import cv2

def readImages():
    print("Reading images")
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f: ### Opens the test images file.
        file_content = f.read() ### Loads the bits from the file into the file_content variable.
    return file_content

def readLabels():
    print("Reading Labels")
    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_lbl = f.read() ### Read in the corresponding 10000 test labels.
    return test_lbl
    
    

def saveImages():
    ### print ("Inside save images")

    file_content = readImages()

    test_lbl = readLabels()

    ### print(type(file_content))

    image = ~np.array(list(file_content[16:800])).reshape(28,28).astype(np.uint8) ### Reshape the list into a 28*28 grid image.

    myInt = int.from_bytes(test_lbl[9:10], byteorder="big") ### Read the next bit into the myInt variable.
    print("The next number is " + str(myInt)) ### Convert the int to a string and output to the screen.
    

    ### Save the images.

    start = 16 ### The first bit of the first image.
    interval = 800 ### The last bit of the first image.
    startLabels = 8 ### The first bit of the first label.
    endLabels = 9 ### The last bit of the first label.
    imgBits = 784 ### The length in bits of each image.
    labelBits = 1 ### The length in bits of each label.
    numberOfItems = int.from_bytes(test_lbl[4:8], byteorder="big") ### the number of items to loop through

    for x in range(5): ### for 60,000 iterations
        image = ~np.array(list(file_content[start:interval])).reshape(28,28).astype(np.uint8) ### assign this slice of bits to the image variable
        myInt = int.from_bytes(test_lbl[startLabels:endLabels], byteorder="big") ### assign this slice of bits to the myInt variable
        start += imgBits ### increase the start pointer by the number of bits per image
        interval += imgBits ### increase the interval pointer by the number of bits per image
        startLabels += labelBits ### increase the startLabels pointer by the number of bits per label
        endLabels += labelBits ### increase the endLabels pointer by the number of bits per label
        
        
        cv2.imwrite(str(myInt) + "_" + str(x) + '.png', image) ### save the i,age variable as a .png file with the name of the corresponding label _ the iteration number for uniqueness

    print("Images saved!")


