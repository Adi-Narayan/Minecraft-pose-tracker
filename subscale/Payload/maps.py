import numpy as np
import cv2 as cv
from sklearn.preprocessing import normalize
from PIL import Image, ImageEnhance, ImageOps

def func_disparity(imgL_path, imgR_path, window_size, minDisparity2, numDisparities2, blockSize2, 
                   disp12MaxDiff2, uniquenessRatio2, speckleWindowSize2, speckleRange2, 
                   preFilterCap2, brightness, contrast, event=None):

    # Load images from the provided paths
    imgL = Image.open(imgL_path)
    imgR = Image.open(imgR_path)
    print(imgL.size)

    # Expand the images by adding a border
    imgL = ImageOps.expand(imgL, border=50)
    imgR = ImageOps.expand(imgR, border=50)    
    
    # Adjust contrast
    contrastL = ImageEnhance.Contrast(imgL)
    contrastR = ImageEnhance.Contrast(imgR)
    imgL = contrastL.enhance(contrast)
    imgR = contrastR.enhance(contrast)

    # Adjust brightness
    brightnessL = ImageEnhance.Brightness(imgL)
    brightnessR = ImageEnhance.Brightness(imgR)
    imgL = brightnessL.enhance(brightness)
    imgR = brightnessR.enhance(brightness)

    # Convert images to grayscale and then to numpy arrays
    imgL = imgL.convert('L')
    imgL = np.array(imgL)
    imgR = imgR.convert('L')
    imgR = np.array(imgR)
    
    # Create the stereo matcher
    left_matcher = cv.StereoSGBM_create(
        minDisparity=minDisparity2,
        numDisparities=numDisparities2,  # max_disp has to be dividable by 16
        blockSize=blockSize2,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=disp12MaxDiff2,
        uniquenessRatio=uniquenessRatio2,
        speckleWindowSize=speckleWindowSize2,
        speckleRange=speckleRange2,
        preFilterCap=preFilterCap2,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )

    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1.0
 
    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    print('computing disparity...')
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

    filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)

    height, width = filteredImg.shape
    filteredImg = np.delete(filteredImg, np.s_[0:50], axis=0)
    filteredImg = np.delete(filteredImg, np.s_[height-100:height-50], axis=0)
    filteredImg = np.delete(filteredImg, np.s_[0:50], axis=1)
    filteredImg = np.delete(filteredImg, np.s_[width-100:width-50], axis=1)
    print(filteredImg.shape)
    
    return filteredImg

# Parameters
window_size = 3
minDisparity2 = 15
numDisparities2 = 16  # max_disp has to be dividable by 16
blockSize2 = 20  # Maybe 20 is optimal
disp12MaxDiff2 = 1
uniquenessRatio2 = 15
speckleWindowSize2 = 0
speckleRange2 = 2
preFilterCap2 = 63
brightness = 1
contrast = 1

# Input image paths
imgL_path = 'Disparity-Map\Stereo Pairs\Pair 2\view1.png'
imgR_path = 'Disparity-Map\Stereo Pairs\Pair 2\view2.png'

# Compute disparity
disparity = func_disparity(imgL_path, imgR_path, window_size, minDisparity2, numDisparities2, blockSize2, 
                           disp12MaxDiff2, uniquenessRatio2, speckleWindowSize2, speckleRange2, 
                           preFilterCap2, brightness, contrast, event=None)

# Display the disparity map
file = Image.fromarray(disparity)
file.show()