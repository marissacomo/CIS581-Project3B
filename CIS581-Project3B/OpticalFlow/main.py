import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import geometric_transform
from scipy import ndimage
from objectTracking import objectTracking

def main(mediaFilePath) :

    rawVideo = cv2.VideoCapture(mediaFilePath)
    objectTracking(rawVideo)


main('Easy.mp4')