__author__ = 'Rahul'

import cv2
import numpy
import os

"""
The module deal with the data in the form of image.
The are four methods:
    1- imageToArray(<image-path>) : it converts each image into an array of dimension rows*columns.
    2- prepImgDataSet(<Dir-path>) : it looks for images in a specific directory and prepares the data set.
    3- arrayToImage(<data-array>, <rows>, <columns>): it converts the image array to an image matrix of shape rows x columns.
    4- showImage(<image-matrix>): it displays the image corresponding to input matrix.
There are three global variable:
    1- dataSet : it holds the array of image data
    2- rows and columns : it informs about the dimension of the matrix of image.
"""
rows= None
columns= None
dataSet= None
#method to read the image and convert it into an array
def imageToArray(image):

    #reading the image in gray scale
    tempImg = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    global rows, columns
    if rows==None and columns==None:
        rows = tempImg.shape[0]
        columns = tempImg.shape[1]

    #detecting number of rows and columns in the image

    row = tempImg.shape[0]
    column = tempImg.shape[1]
    if row==rows and column==columns:
        # reshaping the image into an array
        tempImg= cv2.divide(tempImg.astype(float),255.00)
        imageArray= tempImg.reshape(rows*columns)
       # temp=numpy.divide(imageArray,255)
        return imageArray
    else:
        print "size mismatch not considering the image"
        return None


def prepImgDataSet(pathOfDir):

    # listing the images in the directory
    listing= os.listdir(pathOfDir)

    # consider only image file extentions
    fileXten = ('.png', '.jpg', '.jpeg', '.gif','.pgm')
    dataList=[]
    for file in listing:
        #checking the image file extentions
        if file.endswith(fileXten):


            tempPath= pathOfDir+"/"+file
            tempArray= imageToArray(tempPath)
            if tempArray != None:
                tempArray =tempArray # normalizing the data between [0 and 1]
                dataList.append(tempArray)

    #dataSet = numpy.asarray(dataList)
    dataSet=dataList

    return dataSet, rows, columns

def arrayToImage(arrayData, rows, columns):
    #converting the data into a matrix
    tempMat= arrayData.reshape(rows, columns)
    return tempMat

def showImage(imageMat, scale=1):
    tempRes=cv2.resize(imageMat,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('result', tempRes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def createImage(path, imageMat, scale=1):
    tempRes=cv2.resize(imageMat,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(path,tempRes)
def main(path):
    # call the prepare image data
    global dataSet, rows, columns
    dataSet= prepImgDataSet(path)
    #for each image in the data-set display the image based upon the scale
    for data in dataSet:

        imageData= arrayToImage(data, rows, columns)
        imageData = cv2.multiply(imageData,255) # de-normalizing the data to its original form
        showImage(imageData,1)

    print dataSet
if __name__=="__main__":
    path= '/Users/Rahul/PycharmProjects/turtle/testmnist'
    main(path)