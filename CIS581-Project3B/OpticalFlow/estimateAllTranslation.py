import cv2
import scipy
import numpy as np
import matplotlib.pyplot as plt
from estimateFeatureTranslation import estimateFeatureTranslation
from interp2 import interp2

patchSize = 3
yPatch, xPatch = np.meshgrid(np.arange(-patchSize, patchSize + 1, 1), np.arange(-patchSize, patchSize + 1, 1))
xPatch = xPatch.flatten()
yPatch = yPatch.flatten()

patchWidth = 2 * patchSize + 1
patchHeight = 2 * patchSize + 1

DISTANCE_THRESHOLD = 4

DEBUG = False

def estimateAllTranslation(startXs, startYs, img1, img2, numIterations):

    newXs = np.empty((startXs.shape[0]))
    newYs = np.empty((startYs.shape[0]))
    newXs.fill(-1)
    newYs.fill(-1)

    I1 = None
    I0 = None
    Ix = None
    Iy = None

    validFeatures = np.greater(startXs, -1)
    validX = np.copy(startXs[validFeatures])
    validY = np.copy(startYs[validFeatures])

    originalX = np.copy(startXs[validFeatures])
    originalY = np.copy(startYs[validFeatures])

    if DEBUG:
        print('validX', validX)
        print('validY', validY)

    for j in range(0, numIterations):
        if DEBUG:
            print("--------------- ITERATION --------------- \n")
        
        tiledOffsetsX = np.tile(xPatch, (validX.shape[0], 1))
        tiledOffsetsY = np.tile(yPatch, (validY.shape[0], 1))

        tiledOffsetsX = np.add(tiledOffsetsX, np.tile(np.array([validX]).transpose(), (1, tiledOffsetsX.shape[1])))
        tiledOffsetsY = np.add(tiledOffsetsY, np.tile(np.array([validY]).transpose(), (1, tiledOffsetsX.shape[1])))

        tiledOffsetsX = np.clip(tiledOffsetsX, 0, img1.shape[0] - 1)
        tiledOffsetsY = np.clip(tiledOffsetsY, 0, img1.shape[1] - 1)

        if DEBUG:
            print('tiledOffsetsX', tiledOffsetsX.reshape(patchWidth, patchHeight))
            print('tiledOffsetsY', tiledOffsetsY.reshape(patchWidth, patchHeight))

        I1 = interp2(img2, tiledOffsetsY, tiledOffsetsX)

        if I0 is None:
            I0 = interp2(img1, tiledOffsetsY, tiledOffsetsX)

            tiledMinusOneX = tiledOffsetsX - 1
            tiledMinusOneY = tiledOffsetsY - 1

            tiledMinusOneX = np.clip(tiledMinusOneX, 0, img1.shape[0] - 1)
            tiledMinusOneY = np.clip(tiledMinusOneY, 0, img1.shape[1] - 1)

            if DEBUG:
                print('tiledMinusOneX', tiledMinusOneX.reshape(patchWidth, patchHeight))
                print('tiledMinusOneY', tiledMinusOneY.reshape(patchWidth, patchHeight))


            Ix1 = interp2(img1, tiledOffsetsY, tiledMinusOneX)
            Iy1 = interp2(img1, tiledMinusOneY, tiledOffsetsX)

            Ix = I0 - Ix1
            Iy = I0 - Iy1

            if DEBUG:
                print('Ix1', Ix1.reshape(patchWidth, patchHeight))
                print('Ix', Ix.reshape(patchWidth, patchHeight))
                print('Iy1', Iy1.reshape(patchWidth, patchHeight))
                print('Iy', Iy.reshape(patchWidth, patchHeight))

        It = I1 - I0

        if DEBUG:
            print('I0', I0.reshape(patchWidth, patchHeight))
            print('I1', I1.reshape(patchWidth, patchHeight))
            print('It', It.reshape(patchWidth, patchHeight))
    
        for i in range(0, validX.shape[0]):
            validX[i], validY[i] = estimateFeatureTranslation(validX[i], validY[i], Ix[i], Iy[i], It[i])

        if DEBUG:
            print('validX', validX)
            print('validY', validY)


    for i in range(0, validX.shape[0]):
        dist = ((originalX[i] - validX[i]) * (originalX[i] - validX[i])) + ((originalY[i] - validY[i]) * (originalY[i] - validY[i]))
        
        if dist > DISTANCE_THRESHOLD * DISTANCE_THRESHOLD:
            continue

        if validX[i] >= img1.shape[0]:
            continue

        if validY[i] >= img1.shape[1]:
            continue

        newXs[i] = validX[i]
        newYs[i] = validY[i]

    return newXs, newYs