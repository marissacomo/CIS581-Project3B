import cv2
import numpy as np
import matplotlib.pyplot as plt

DEBUG = False

def estimateFeatureTranslation(startX, startY, Ix, Iy, It):

    IxIx = np.dot(Ix, Ix)
    IyIy = np.dot(Iy, Iy)
    IxIy = np.dot(Ix, Iy)

    IxIt = -np.dot(Ix, It)
    IyIt = -np.dot(Iy, It)

    a = np.array([[IxIx, IxIy], [IxIy, IyIy]])
    b = np.array([IxIt, IyIt])

    x = np.linalg.solve(a, b)

    if DEBUG:
        print('Delta: ', x)

    return startX + x[0], startY + x[1]