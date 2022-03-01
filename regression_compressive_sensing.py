# This is for ECE580: Intro to machine learning Spring 2020 in Duke
# This is translated to Python from show_chanWeights.m file provided by Prof. Li by 580 TAs

# import ext libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn import linear_model as lm
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy import signal
# from scipy.misc import imread   # Make Sure you install the required packages like Pillow and scipy

def imgRead(fileName):
    """
    load the input image into a matrix
    :param fileName: name of the input file
    :return: a matrix of the input image
    Examples: imgIn = imgRead('lena.bmp')
    """
    imgIn = plt.imread(fileName)
    return imgIn

def imgShow(imgOut, title):
    """
    show the image saved in a matrix
    :param imgOut: a matrix containing the image to show
    :return: None
    """
    imgOut = np.uint8(imgOut)
    plt.figure()
    plt.imshow(imgOut, cmap='gray', vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")

def imgRecover(imgIn, blkSize, numSample):
    """
    Recover the input image from a small size samples
    :param imgIn: input image
    :param blkSize: block size
    :param numSample: how many samples in each block
    :return: recovered image
    """
    imgOut = np.zeros(imgIn.shape)
    sampledImg = np.zeros(imgIn.shape)
    blocks = splitImgIntoBlks(imgIn, blkSize)
    T = getFullTMatrix(blkSize)
    for i, rowOfBlocks in enumerate(blocks):
        for j, block in enumerate(rowOfBlocks):
            C = np.reshape(block, block.size)

            B, A, sampledC = sampleCandT(C, T, numSample)
            regCoef = getLambdaByCrossValidation(B, A)
            #regCoef = 0.000001
            model = getModel(B, A, regCoef)
            gamma = model.coef_
            intercept = model.intercept_

            newBlock = reconstructBlk(sampledC, T, gamma, intercept)

            imgOut[i*blkSize:(i+1)*blkSize, j*blkSize:(j+1)*blkSize] = newBlock
            sampledBlk = np.reshape(sampledC, (blkSize, blkSize))
            sampledImg[i*blkSize:(i+1)*blkSize, j*blkSize:(j+1)*blkSize] = sampledBlk

    return imgOut, sampledImg

def splitImgIntoBlks(imgIn, blkSize):
    """Split an image into blocks with the correct size.
    Adapted from https://www.reddit.com/r/learnpython/comments/nj7ptv/efficiently_divide_numpy_array_into_blocks_sized/"""

    B = imgIn.reshape((-1, blkSize, imgIn.shape[1]//blkSize, blkSize))
    C = B.transpose((0,2,1,3))
    return C

def sampleCandT(C, T, numSample):
    """Samples pixels and DCTs based on numSamples"""
    n = min(numSample, C.size)
    sampledIndeces = set(random.sample(range(C.size), n))
    sampledC = np.full(C.shape, -1)
    B = np.empty((n))
    A = np.empty((n, T[0].size))

    for i, index in enumerate(sampledIndeces):
        B[i] = C[index]
        A[i] = T[index]
        sampledC[index] = C[index]

    return B, A, sampledC

@ignore_warnings(category=ConvergenceWarning)
def getModel(B, A, regCoef):
    """Determines the DCT Coefficients for the 2-D DCT
    Uses sklearn LASSO model: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html"""
    clf = lm.Lasso(alpha=regCoef, fit_intercept=True)
    #clf = lm.LinearRegression(fit_intercept=True)
    clf.fit(A, B)
    return clf

def getLambdaByCrossValidation(B, A):
    """Determines the optimal regularization parameter lambda via cross validation"""
    n = 20
    lambdas = np.logspace(-6, 6, n)
    maxError = float("inf")
    bestLambda = 1
    for l in lambdas:
        trainPixels, newA, testIndeces = getSubsets(B, A, int(np.floor(B.size/6)))
        clf = getModel(trainPixels, newA, l)
        prediction = clf.predict(A)
        error = getMSECrossValidation(prediction, B, testIndeces)
        error = getMSECrossValidation2(prediction, B)
        if error < maxError:
            maxError = error
            bestLambda = l
    
    return bestLambda

def getMSECrossValidation(prediction, B, testIndeces):
    error = 0
    for index in testIndeces:
        error += (B[index] - prediction[index])**2
    return (1/len(testIndeces)) * error 

def getMSECrossValidation2(predictions, B):
    error = 0
    for index, pred in enumerate(predictions):
        error += (B[index] - pred)**2
    return error/len(predictions)


def getSubsets(B, A, m):
    """Returns a subset of m testing pixels from the sampled pixels. 
    Remaining pixels are used as training pixels"""
    sampledIndeces = set(random.sample(range(B.size), m))
    trainSet = np.empty((B.size-m))
    newA = np.empty((B.size - m, A[0].size))
    k=0
    for i in range(B.size):
        if i not in sampledIndeces:
            trainSet[k] = B[i]
            newA[k] = A[i]
            k+=1
    return trainSet, newA, sampledIndeces


def getFullTMatrix(blkSize):
    """Finds the full T matrix (All DCT Basis functions)
    Removes the first column of the T matrix (that will be handled by the intercept term of the LASSO"""
    P = blkSize
    Q = blkSize
    T = np.zeros((P*Q, P*Q))
    for v in range(1,Q+1):
        for u in range(1,P+1):
            for y in range(1, Q+1):
                for x in range(1,P+1):
                    element = 0
                    a = np.sqrt(2/P)
                    b = np.sqrt(2/Q)
                    if u == 1:
                        a = np.sqrt(1/P)
                    if v == 1:
                        b = np.sqrt(1/Q)
                    element = a * b * np.cos((np.pi*(2*x-1)*(u-1))/(2*P)) * np.cos((np.pi*(2*y-1)*(v-1))/(2*Q))
                    T[(x-1)+(y-1)*Q][(u-1)+(v-1)*Q] = element
    return T[:,1:] # Remove DC term from T matrix

def reconstructBlk(sampledC, T, gamma, intercept):
    C = np.matmul(T, gamma) + intercept
    for i in range(sampledC.size):
        if sampledC[i] != -1:
            C[i] = sampledC[i]
    blkSize = int(np.sqrt(C.size))
    return np.reshape(C, (blkSize, blkSize))

def imgFilter(img, threshold):
    filteredImg = signal.medfilt2d(img, 3)
    """
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if not (filteredImg[i][j] < pixel - threshold or filteredImg[i][j] > pixel + threshold):
                filteredImg[i][j] = pixel
    """

    return filteredImg

def imgMSE(originalImg, newImg):
    error = 0
    for i, row in enumerate(originalImg):
        for j, pixel in enumerate(row):
            error += (pixel - newImg[i][j])**2
    return (1/originalImg.size) * error

def plotErrors(sampleSizes, recoveredErrors, filteredErrors, name):
    plt.close()
    plt.figure()
    plt.plot(sampleSizes, recoveredErrors, label="Recovered Image")
    plt.plot(sampleSizes, filteredErrors, label="Filtered Image")
    plt.title(f"MSE for Recovered and Filtered {name} Image")
    plt.xlabel('Sample Size')
    plt.ylabel('MSE')
    plt.legend()


def runImgCompression(source, sampleSizes, blockSize, name, threshold):
    recoveredErrors = []
    filteredErrors = []
    for numSamples in sampleSizes:
        recovered, sampled = imgRecover(source, blockSize, numSamples)
        imgShow(sampled, title=f"{name} Image with {numSamples} Samples per Block")
        plt.savefig("report/images/" + name.lower() + "/sampled" + str(numSamples) + ".png")
        imgShow(recovered, title=f"Recovered {name} Image for {numSamples} Samples per Block")
        plt.savefig("report/images/" + name.lower() + "/recovered" + str(numSamples) + ".png")
        filtered = imgFilter(recovered, threshold)
        imgShow(filtered, title=f"Median Filtered {name} Image for {numSamples} Samples per Block")
        plt.savefig("report/images/" + name.lower() + "/filtered" + str(numSamples) + ".png")
        recoveredError = imgMSE(source, recovered)
        filteredError = imgMSE(source, filtered)
        print(f"MSE for recovered image for {numSamples} Samples per Block: {recoveredError}")
        print(f"MSE for filtered image for {numSamples} Samples per Block: {filteredError}")
        recoveredErrors.append(recoveredError)
        filteredErrors.append(filteredError)

    plotErrors(sampleSizes, recoveredErrors, filteredErrors, name)


if __name__ == '__main__':
    boatSource = imgRead('fishing_boat.bmp')
    boatSamples = [10, 20, 30, 40, 50]
    boatBlockSize = 8
    #runImgCompression(boatSource, boatSamples, boatBlockSize, "Boat", threshold=25)

    #plt.show()

    natureSource = imgRead('nature.bmp')
    natureSamples = [10, 30, 50, 100, 150]
    natureBlockSize = 16
    runImgCompression(natureSource, natureSamples, natureBlockSize, "Nature", threshold=50)

    plt.show()
