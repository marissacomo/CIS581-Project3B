import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.ndimage import geometric_transform
from scipy import ndimage
from corner_detector import corner_detector

def getFeatures(gray, bbox, N):
    # fig, axes = plt.subplots(1, 1, gridspec_kw = {'width_ratios':[gray.shape[1]]})
    # axes.imshow(gray, cmap='gray')
    # axes.axis('off')
    # axes.set_title('Image : [' + str(gray.shape[1]) + ', ' + str(gray.shape[0]) + ']')
    # fig.tight_layout()
    
    x_coords = np.empty((bbox.shape[0], N))
    y_coords = np.empty((bbox.shape[0], N))

    x_coords.fill(-1)
    y_coords.fill(-1)

    for i in range(0, bbox.shape[0]):
        minX = bbox[i, 0, 0]
        minY = bbox[i, 0, 1]
        maxX = bbox[i, 1, 0]
        maxY = bbox[i, 1, 1]
        inBox = np.array(gray)[minX : maxX, minY : maxY]
        cornerStrength = corner_detector(inBox, N)

        # rect = patches.Rectangle((minY, minX), maxY - minY, maxX - minX, linewidth=1, edgecolor='r', facecolor='none')
        # axes.add_patch(rect)

        count = 0
        for j in cornerStrength:
            y, x = j.ravel()
            x_coords[i, count] = minX + x
            y_coords[i, count] = minY + y

            count += 1

            # circle = plt.Circle((minY + y, minX + x), 5, color='r', linewidth=1, fill=False)
            # axes.add_artist(circle)



    # plt.show()

    return x_coords, y_coords
