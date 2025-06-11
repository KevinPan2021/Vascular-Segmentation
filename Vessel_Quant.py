import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2



# vessel area density (VAD)
# vessel skeleton density (VSD)
# vessel diameter index (VDI)
# vessel perimeter index (VPI)
# vessel complexity index (VCI)
class Vessel_Quantification():
    def __init__(self, img, vessel_mask, ring_mask, FOV):
        # parameters
        self.FOV = FOV
        self.window_size = 35
        self.Ibinary = vessel_mask
        self.Ibinary2 = vessel_mask
        
        # get Inner_Ring, Outer_Ring from ring_mask
        self.ring_mask = ring_mask
        self.inner_ring_mask = None
        self.outer_ring_mask = None
        
        if not ring_mask is None:
            # Find contours with hierarchy info
            contours, hierarchy = cv2.findContours(self.ring_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            
            # Separate outer and inner contours using hierarchy
            radii_centers = []
            for idx, cnt in enumerate(contours):
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                radii_centers.append((radius, (x, y)))
                
            # Sort contours by radius
            radii_centers.sort(key=lambda x: x[0])
            (inner_radius, center1), (outer_radius, center2) = radii_centers
            
            # Average centers for robustness
            center_x = (center1[0] + center2[0]) / 2
            center_y = (center1[1] + center2[1]) / 2
            center = (int(center_x), int(center_y))
            
            h, w = self.ring_mask.shape
            self.inner_ring_mask = np.zeros((h, w), dtype=np.uint8)
            self.outer_ring_mask = np.zeros((h, w), dtype=np.uint8)
        
            # Draw filled circles
            cv2.circle(self.inner_ring_mask, center, int(inner_radius), 255, -1)
            cv2.circle(self.outer_ring_mask, center, int(outer_radius), 255, -1)
        
        
        '''
        import matplotlib.pyplot as plt
        
        plt.figure()
        plt.imshow(self.inner_ring_mask)
        
        plt.figure()
        plt.imshow(self.outer_ring_mask)
        
        
        plt.figure()
        plt.imshow(self.ring_mask)
        
        plt.show()
        '''
        
        # initialize placeholders
        self.skeleton = None
        self.vad_map = None
        self.skeleton_map = None
        self.vesselper = None
    
    
    @staticmethod
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


    # get the inner, ring, outer, and total metrics
    def get_ring_metrics(self, img):
        # get the total metric
        nonzero_values = img[img != 0]
        total_metric = np.nanmean(nonzero_values) if len(nonzero_values) > 0 else 0
        
        # get the inner metric
        inner_img = img.copy()
        inner_metric = total_metric
        if not self.inner_ring_mask is None:
            inner_img *= self.inner_ring_mask.astype(bool)
            nonzero_values = inner_img[inner_img != 0]
            inner_metric = np.nanmean(nonzero_values) if len(nonzero_values) > 0 else 0
            
        # get the ring metric
        ring_img = img.copy()
        ring_metric = total_metric
        if not self.ring_mask is None:
            ring_img *= self.ring_mask.astype(bool)
            nonzero_values = ring_img[ring_img != 0]
            ring_metric = np.nanmean(nonzero_values) if len(nonzero_values) > 0 else 0
        
        # get the outer metric
        outer_img = img.copy()
        outer_metric = total_metric
        if not self.outer_ring_mask is None:
            outer_img *= self.ring_mask.astype(bool)
            nonzero_values = outer_img[outer_img != 0]
            outer_metric = np.nanmean(nonzero_values) if len(nonzero_values) > 0 else 0
        
        return ring_img, [inner_metric, ring_metric, outer_metric, total_metric]
        
        
    # Vessel skeleton map. Obtained by iteratively deleting the pixels in the outer boundary of the vessel 
    # area map until one pixel remained along the width direction of the vessels
    def calc_skeleton(self, ):
        # # Skeletonize the binary image
        self.skeleton = self.thinning_zhang_suen(self.Ibinary2)        
        self.skeleton = self.skeleton.astype(bool)
        skeleton = self.skeleton.copy()
        if not self.ring_mask is None:
            skeleton *= self.ring_mask.astype(bool)
        return skeleton
    
    
    # vessel skeleton density (VSD)
    def calc_skeleton_map(self, ): 
        vsd = self.skeleton
        window_size = self.window_size
        
        pad_height = (window_size - 1) // 2
        pad_width = (window_size - 1) // 2
        vsd = np.pad(vsd, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
        
        # Create sliding windows over the image
        windows = self.view_as_windows(vsd, window_size)

        # Calculate the white pixel density for each window
        white_pixel_count = np.sum(windows, axis=(2, 3))
        self.vsd_map = white_pixel_count / (window_size * window_size)
        
        self.vsd_map = cv2.GaussianBlur(self.vsd_map, (15, 15), 0)
        skeleMap = self.vsd_map * self.Ibinary2
        
        return self.get_ring_metrics(skeleMap)


    # vessel area density (VAD)
    # VAD is calculated as a unit less ratio of the total image area occupied by the vasculature to the 
    # total image area in the binary vessel maps
    def calc_area_map(self, ):
        vad = self.Ibinary
        window_size = self.window_size
        
        pad_height = (window_size - 1) // 2
        pad_width = (window_size - 1) // 2
        vad = np.pad(vad, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
        
        # Create sliding windows over the image
        windows = self.view_as_windows(vad, window_size)

        # Calculate the white pixel density for each window
        white_pixel_count = np.sum(windows, axis=(2, 3))
        self.vad_map = white_pixel_count / (window_size * window_size)
        
        areaMap = self.vad_map * self.Ibinary2
        
        return self.get_ring_metrics(areaMap)


    

    # vessel diameter index (VDI)
    # VDI is calculated using both the vessel area map and the skeletonized vessel map to yield the averaged vessel caliber
    def calc_diameter_map(self, ):
        vad = self.Ibinary2
        vsd = self.skeleton
        window_size = self.window_size
        
        pad_height = (window_size - 1) // 2
        pad_width = (window_size - 1) // 2
        vad = np.pad(vad, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
        vsd = np.pad(vsd, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
        
        # Create sliding windows over the image
        windows_1 = self.view_as_windows(vad, window_size)
        windows_2 = self.view_as_windows(vsd, window_size)
    
        # Calculate the white pixel density for each window
        white_pixel_count_1 = np.sum(windows_1, axis=(2, 3))
        white_pixel_count_2 = np.sum(windows_2, axis=(2, 3))
        
        # Calculate VDI for each window (avoid division by zero)
        self.vdi_map = np.where(white_pixel_count_2 == 0, 0, white_pixel_count_1 / (white_pixel_count_2 + np.finfo(np.float64).eps) )
        self.vdi_map = cv2.GaussianBlur(self.vdi_map, (15, 15), 0)
        diameter = self.vdi_map*self.Ibinary2
        
        # convert from pixels to um
        height, width = diameter.shape
        diameter = diameter / height * self.FOV
        
        return self.get_ring_metrics(diameter)

    

    # Vessel perimeter map
    # which is obtain by detecting the edge of vessels in the vessel area map and deleting pixels that are not on the edge of vessels.
    def calc_perimeter_map(self, ):
        Ibinary2 = self.Ibinary2.astype(np.uint8)
        sobel_x = np.uint8(np.absolute(cv2.Sobel(Ibinary2,cv2.CV_64F,1,0))) #sobel_x
        sobel_y = np.uint8(np.absolute(cv2.Sobel(Ibinary2,cv2.CV_64F,0,1))) #sobel_y
        edges = cv2.bitwise_or(sobel_x, sobel_y)

        # Delete pixels that are not on the edge of vessels
        vpi = self.Ibinary2 * edges
        
        window_size = self.window_size
        
        # # Convert binary image to float and normalize it
        self.vpi = vpi.astype(np.float32)
        
        pad_height = (window_size - 1) // 2
        pad_width = (window_size - 1) // 2
        vpi = np.pad(vpi, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
        
    
        # # Create sliding windows over the image
        windows = self.view_as_windows(vpi, window_size)

        # # Calculate the white pixel density for each window
        white_pixel_count = np.sum(windows, axis=(2, 3))
        self.vpi_map = white_pixel_count / (window_size * window_size)
        self.vpi_map = cv2.GaussianBlur(self.vpi_map, (15, 15), 0)
        
        #self.vpi_map = self.vdi_map*np.pi
        
        periMap = self.vpi_map*self.Ibinary2
        
        return self.get_ring_metrics(periMap)

    

    # VCI is also a unit less parameter and derived from the digital image processing discipline. With the vessel complexity map, 
    #specific locations where the vascular morphology is less complicated could be identified.
    def calc_complexity_map(self, ):     
        vpi = self.vpi_map.astype(np.float32)
        window_size = self.window_size
        
        pad_height = (window_size - 1) // 2
        pad_width = (window_size - 1) // 2
        vpi = np.pad(vpi, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
        
        # Create sliding windows over the image
        windows = self.view_as_windows(vpi, window_size)

        # Calculate the white pixel density for each window
        white_pixel_count = np.sum(windows, axis=(2, 3))
        self.vci_map = white_pixel_count / (window_size * window_size)
        
        self.vci_map = cv2.GaussianBlur(self.vci_map, (15, 15), 0)
        
        compMap = self.vci_map * self.Ibinary2
        
        # Calculate total mean VSD
        return self.get_ring_metrics(compMap)
