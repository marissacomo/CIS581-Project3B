import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import geometric_transform
from scipy import ndimage
from corner_detector import corner_detector
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation

import skimage

SHOW_FRAME = False

def objectTracking(rawVideo) :

    maxFeaturePoints = 10
    numIterations = 8

    if(not rawVideo.isOpened()):
        print('Failed!!!')
        return

    # Video Easy
    bbox = np.array([
        [[180, 288], [268, 406]], # Car
        # [[203, 300], [216, 312]], # Test
        # [[126, 160], [160, 182]], # Person
    ])

    # # Video Medium
    # bbox = np.array([
    #     [[182, 457], [276, 520]], # Car
    #     # [[203, 300], [216, 312]], # Test
    # ])

    ret, prevFrame = rawVideo.read()   
    prevGray = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)

    x_features, y_features = getFeatures(prevGray, bbox, maxFeaturePoints)

    prevGray = np.array(prevGray)

    frameCount = 1

    while(rawVideo.isOpened()):
        ret, nextFrame = rawVideo.read()

        nextGray = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)
        nextGray = np.array(nextGray)

        if SHOW_FRAME:
            fig, axes = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[prevGray.shape[1], prevGray.shape[1]]})
            axes[0].imshow(prevGray, cmap='gray')
            axes[0].axis('off')
            axes[0].set_title('Frame' + str(frameCount - 1) + ': [' + str(prevGray.shape[1]) + ', ' + str(prevGray.shape[0]) + ']')

            axes[1].imshow(nextGray, cmap='gray')
            axes[1].axis('off')
            axes[1].set_title('Frame' + str(frameCount) + ': [' + str(nextGray.shape[1]) + ', ' + str(nextGray.shape[0]) + ']')

        newXs = []
        newYs = []

        for i in range(0, bbox.shape[0]):
            newXs, newYs = estimateAllTranslation(x_features[i], y_features[i], prevGray, nextGray, numIterations)

            validFeatures = np.greater(newXs, -1)
            oldValidX = np.copy(x_features[i][validFeatures])
            oldValidY = np.copy(y_features[i][validFeatures])
            newValidX = np.copy(newXs[validFeatures])
            newValidY = np.copy(newYs[validFeatures])

            for j in range(0, newValidX.shape[0]):
                cv2.circle(nextFrame, (int(newValidY[j]), int(newValidX[j])), 4, (0,0,255), thickness=1, lineType=8, shift=0)

                if SHOW_FRAME:
                    circle = plt.Circle((oldValidY[j], oldValidX[j]), 4, color='r', linewidth=1, fill=True)
                    axes[0].add_artist(circle)
                    circle = plt.Circle((newValidY[j], newValidX[j]), 4, color='b', linewidth=1, fill=True)
                    axes[0].add_artist(circle)

                    circle = plt.Circle((newValidY[j], newValidX[j]), 4, color='r', linewidth=1, fill=True)
                    axes[1].add_artist(circle)


            if newValidX.shape[0] >= 2:
                maxX = newValidX[np.argmax(newValidX)]
                maxY = newValidY[np.argmax(newValidY)]
                minX = newValidX[np.argmin(newValidX)]
                minY = newValidY[np.argmin(newValidY)]

                oldMaxX = oldValidX[np.argmax(oldValidX)]
                oldMaxY = oldValidY[np.argmax(oldValidY)]
                oldMinX = oldValidX[np.argmin(oldValidX)]
                oldMinY = oldValidY[np.argmin(oldValidY)]

                scaleX = (maxX - minX) / (oldMaxX - oldMinX)
                scaleY = (maxY - minY) / (oldMaxY - oldMinY)

                meanX = np.mean(newValidX)
                meanY = np.mean(newValidY)

                oldMeanX = np.mean(oldValidX)
                oldMeanY = np.mean(oldValidY)

                translateX = meanX - oldMeanX
                translateY = meanY - oldMeanY

                npMatrix = np.dot(np.array([[scaleX, 0, meanX], [0, scaleY, meanY], [0, 0, 1]]), np.array([[1, 0, -oldMeanX], [0, 1, -oldMeanY], [0, 0, 1]]))

                # transform = skimage.transform.estimate_transform('similarity', np.column_stack([oldValidX, oldValidY]), np.column_stack([newValidX, newValidY]))
                # npMatrix = np.array(transform.params)

                box = bbox[i]

                # minPoint = np.array(npMatrix.dot(np.array([box[0, 0], box[0, 1], 1])))[0]
                # maxPoint = np.array(npMatrix.dot(np.array([box[1, 0], box[1, 1], 1])))[0]
                # cv2.circle(nextFrame, (int(minPoint[1]), int(minPoint[0])), 4, (0,255,0), thickness=1, lineType=8, shift=0)
                # cv2.circle(nextFrame, (int(maxPoint[1]), int(maxPoint[0])), 4, (255,255,0), thickness=1, lineType=8, shift=0)

                boundMin = npMatrix.dot([box[0, 0], box[0, 1], 1])
                boundMax = npMatrix.dot([box[1, 0], box[1, 1], 1])

                # for k in range(0, oldValidX.shape[0]):
                #     transformedPoint = np.array(npMatrix.dot(np.array([oldValidX[k], oldValidY[k], 1])))[0]
                    # cv2.circle(nextFrame, (int(transformedPoint[1]), int(transformedPoint[0])), 4, (0,255,0), thickness=1, lineType=8, shift=0)

                box[0] = np.array([boundMin[0], boundMin[1]])
                box[1] = np.array([boundMax[0], boundMax[1]])

                cv2.rectangle(nextFrame,(box[0, 1], box[0, 0]), (box[1, 1], box[1, 0]), (0, 255, 0), 1)
                bbox[i] = box

            x_features[i] = newXs
            y_features[i] = newYs

        if SHOW_FRAME:
            plt.show()

        # prevFrame = nextFrame
        prevGray = nextGray
        frameCount += 1

        cv2.imshow('frame', nextFrame)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    rawVideo.release()
    cv2.destroyAllWindows()

