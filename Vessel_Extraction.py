import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
from scipy import ndimage



def view_as_windows(arr, window_shape, step=1):
    if isinstance(window_shape, int):
        window_shape = (window_shape, window_shape)
    if isinstance(step, int):
        step = (step, step)
    
    arr_shape = np.array(arr.shape)
    window_shape = np.array(window_shape)
    step = np.array(step)
    
    # Calculate the shape of the output array
    out_shape = ((arr_shape - window_shape) // step) + 1
    out_shape = tuple(out_shape) + tuple(window_shape)
    
    # Calculate the strides for the output array
    strides = tuple(np.array(arr.strides) * step) + arr.strides
    
    # Create the strided view of the input array
    windows = as_strided(arr, shape=out_shape, strides=strides)
    return windows



class Vessel_Quantification():
    def __init__(self, img, thres):
        self.thres = thres

        # apply bilaterial filter to reduce noise
        filtered_img = cv2.bilateralFilter(img, d=3, sigmaColor=75, sigmaSpace=75)

        # min max norm to [0, 255]
        filtered_img = cv2.normalize(filtered_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # CLAHE 
        clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(5,5))
        filtered_img = clahe.apply(filtered_img)

        # create intensity masking
        self.Ibinary = filtered_img > self.thres * 255

        # Perform dilation
        self.Ibinary = cv2.dilate(self.Ibinary.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1)

        # vessel filtering
        Ivessel = self.multiscale_segmentation(img, [3], 10)
        Ivessel = Ivessel > 0.5
        
        # vessel with intensity masking
        self.Ibinary2 = Ivessel * self.Ibinary
        self.Ibinary = Ivessel

        self.window_size = 35

    
    
    # Frangi vessel enhancement function
    # Vessel area map, a binarized vasculature image using hessian filter and adaptive threshold
    def frangi_vessel_enhancement(self, image, sigma=1.0, alpha=0.5, beta=0.5, gamma=15.0):
        # Step 1: Compute the Hessian matrix
        image = image.astype(np.float32)
        Hxx = ndimage.gaussian_filter(image, sigma=sigma, order=(0, 2), mode='reflect')
        Hyy = ndimage.gaussian_filter(image, sigma=sigma, order=(2, 0), mode='reflect')
        Hxy = ndimage.gaussian_filter(image, sigma=sigma, order=(1, 1), mode='reflect')
    
        # Step 2: Compute the eigenvalues and eigenvectors of the Hessian matrix
        lambda1 = 0.5 * (Hxx + Hyy + np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy ** 2))
        lambda2 = 0.5 * (Hxx + Hyy - np.sqrt((Hxx - Hyy) ** 2 + 4 * Hxy ** 2))
    
        # Step 3: Compute vesselness measure using the Frangi's vesselness formula
        Rb = lambda2 / (lambda1 + 1e-10)
        S = np.sqrt(lambda1 ** 2 + lambda2 ** 2)
        V = np.exp(-(Rb ** 2) / (2 * alpha ** 2)) * (1 - np.exp(-(S ** 2) / (2 * beta ** 2)))
        vesselness = np.exp(-(gamma ** 2) / (2 * S ** 2 + np.finfo(np.float64).eps)) * (1 - V)
    
        # Normalize the vesselness to [0, 1]
        vesselness = (vesselness - np.min(vesselness)) / (np.max(vesselness) - np.min(vesselness))
        
        return vesselness
    
    
    def multiscale_segmentation(self, image, scales, otsu_threshold=20):
        grayImg = image
        segmented_images = []
    
        for scale in scales:
            blurImg = cv2.GaussianBlur(grayImg, (scale, scale), 0)
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(7, 7))
            claheImg = clahe.apply(blurImg)
            ret, th = cv2.threshold(claheImg, otsu_threshold, 255, cv2.THRESH_OTSU)
            segmented_images.append(th)
    
        frangi_image = self.frangi_vessel_enhancement(grayImg, sigma=1.0, alpha=0.5, beta=0.5, gamma=15.0)
        combined_segmentation = np.maximum.reduce(segmented_images + [frangi_image])
        return combined_segmentation


    def thinning_zhang_suen(self,image):
        skel = np.zeros(image.shape, np.uint8)
        image = image.astype(np.uint8)
        ret, image = cv2.threshold(image, 0, 255, 0)
    
        while True:
            eroded = cv2.erode(image, None)
            temp = cv2.dilate(eroded, None)
            temp = cv2.subtract(image, temp)
            skel = cv2.bitwise_or(skel, temp)
            image = eroded.copy()
    
            if cv2.countNonZero(image) == 0:
                break
    
        return skel

    
    
