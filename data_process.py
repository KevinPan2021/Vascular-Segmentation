import numpy as np
import torch
import torch.nn.functional as F
from skimage.filters import threshold_otsu


from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSignal, Qt, QPointF, QEvent, QSize, QThread
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QLineEdit, QDesktopWidget, QFileDialog, QCheckBox, QTableWidgetItem


class Data():
    def __init__(self, filename, enface=None, OCT=None, OCTA=None, cavf_pred_2D=None, prediction=None, model_output=None):
        self.filename = filename
        self.enface = enface
        self.OCT = OCT
        self.OCTA = OCTA
        self.cavf_pred_2D = cavf_pred_2D
        self.prediction = prediction
        self.model_output = model_output
        

class ProcessThread(QThread):
    process = pyqtSignal(int)  # Signal to update progress bar
    work_complete_signal = pyqtSignal()  # Signal when work is finished

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True  # Flag to control stopping

    def run(self):
        total_images = len(self.parent.image_list)
        
        for i, data_cls in enumerate(self.parent.image_list):
            if not self.running:
                break  # Stop processing if stopped

            enface = data_cls.enface
            OCT = data_cls.OCT
            OCTA = data_cls.OCTA
            
            data, proj_map = process_data(OCT, OCTA, 'cuda', use_proj_map=True, OCTA_proj_map=enface)
            self.parent.image_list[i].cavf_pred_2D = self.parent.inference(data, proj_map)
            self.parent.image_list[i].prediction = get_cavf_Sparse_RGBA(self.parent.image_list[i].cavf_pred_2D)
            self.parent.image_list[i].model_output = get_cavf_RGB(self.parent.image_list[i].cavf_pred_2D)
            
            # Update progress bar
            progress = int((i + 1) / total_images * 100)
            self.process.emit(progress)
        
        self.work_complete_signal.emit()  # Signal that work is done

    def stop(self):
        self.running = False  # Set flag to stop processing



        
# convert image into RGB
# image should have 3 dimensions (channel, weight, height)
def get_cavf_RGB(image):
    RGB_img = np.zeros((image.shape[1], image.shape[2], 3))

    RGB_img[:, :, 0] = 15 * image[0, :, :] + 171 * image[1, :, :] + 215 * image[2, :, :] + 43 * image[3, :, :] + 166 * image[4, :, :]
    RGB_img[:, :, 1] = 32 * image[0, :, :] + 165 * image[1, :, :] + 25 * image[2, :, :] + 131 * image[3, :, :] + 217 * image[4, :, :]
    RGB_img[:, :, 2] = 53 * image[0, :, :] + 143 * image[1, :, :] + 28 * image[2, :, :] + 186 * image[3, :, :] + 106 * image[4, :, :]
    
    return RGB_img.astype(np.uint8)


def get_cavf_RGBA(image):
    RGBA_img = np.zeros((image.shape[1], image.shape[2], 4))

    RGBA_img[..., 0] = 15 * image[0, ...] + 171 * image[1, ...] + 215 * image[2, ...] +  43 * image[3, ...] + 166 * image[4, ...]
    RGBA_img[..., 1] = 32 * image[0, ...] + 165 * image[1, ...] +  25 * image[2, ...] + 131 * image[3, ...] + 217 * image[4, ...]
    RGBA_img[..., 2] = 53 * image[0, ...] + 143 * image[1, ...] +  28 * image[2, ...] + 186 * image[3, ...] + 106 * image[4, ...]
    
    # extract the background and the capillaries
    background = np.argmax(image, axis=0) == 0
    capillaries = np.argmax(image, axis=0) == 1
    mask = np.logical_or(background, capillaries)  # Boolean mask
    
    # mask the backgorund
    RGBA_img[..., 3] = 255
    
    # Apply mask to set RGB and alpha to 0
    RGBA_img[mask] = [0, 0, 0, 0]
    
    
    return RGBA_img.astype(np.uint8)


def get_cavf_Sparse_RGBA(image):
    RGBA_img = np.zeros((image.shape[1], image.shape[2], 4))

    RGBA_img[..., 0] = 15 * image[0, ...] + 171 * image[1, ...] + 215 * image[2, ...] +  43 * image[3, ...] + 166 * image[4, ...]
    RGBA_img[..., 1] = 32 * image[0, ...] + 165 * image[1, ...] +  25 * image[2, ...] + 131 * image[3, ...] + 217 * image[4, ...]
    RGBA_img[..., 2] = 53 * image[0, ...] + 143 * image[1, ...] +  28 * image[2, ...] + 186 * image[3, ...] + 106 * image[4, ...]
    
    # extract the background and the capillaries
    background = np.argmax(image, axis=0) == 0
    capillaries = np.argmax(image, axis=0) == 1
    mask = np.logical_or(background, capillaries)  # Boolean mask
    
    # mask the backgorund
    RGBA_img[..., 3] = 255
    
    # Apply mask to set RGB and alpha to 0
    RGBA_img[mask] = [0, 0, 0, 0]
    
    
    # Separate RGB and Alpha channels
    rgb = RGBA_img[..., :3]  # Extract RGB channels
    alpha = RGBA_img[..., 3]  # Extract Alpha channel

    # Get the index of the max channel per pixel (ignoring Alpha)
    max_indices = np.argmax(rgb, axis=2)

    # Create an empty RGB array (same shape as input RGB)
    sparse_rgb = np.zeros_like(rgb)

    # Set the max channel to 255
    for i in range(3):  # Iterate over R, G, B channels
        sparse_rgb[..., i] = (max_indices == i) * 255  # Set max channel to 255

    # Combine sparse RGB with original Alpha channel
    sparse_image = np.dstack((sparse_rgb, alpha))  # Stack along last axis to restore RGBA format

    return sparse_image.astype(np.uint8)


# Convert QPixmap to a NumPy array
def qpixmap_to_numpy(pixmap):
    qimage = pixmap.toImage()  # Convert QPixmap to QImage
    qimage = qimage.convertToFormat(QImage.Format_RGBA8888)  # Ensure 4-channel format

    width, height = qimage.width(), qimage.height()
    ptr = qimage.bits()  # Get memory pointer
    ptr.setsize(qimage.byteCount())  # Define memory size

    # Convert to NumPy array (H, W, 4) for RGBA
    arr = np.array(ptr, dtype=np.uint8).reshape((height, width, 4))
    return arr



def normalize(img):
    if type(img) == torch.Tensor:
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    elif type(img) == np.ndarray:
        return (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        raise ValueError("Input image must be a numpy array or a torch tensor.")
        



def get_ROI(img, target_depth=550):
    """
        Get the region of interest (ROI) of the input image 
        
        Args:
            - img: 5D tensor of shape (1, 2, D, H, W). where the first channel is OCT and the second channel is OCTA
            - target_depth: target depth of the ROI. 
                            Assumes target_depth will be larger than calculated otsu threshold roi depth. 
            
            - Returns:
                - 5D tensor of shape (1, 2, roi_depth, H, W). Note that roi_depth may not be equal to target_depth 
                    if target depth is smaller than the calculated roi depth.
    """
    assert img.ndim == 5, "Input image must be 5D = (B, C, D, H, W)."
    assert img.shape[1] == 2, "Input image must have 2 channels, with the first channel being OCT and the second channel being OCTA."
    
    
    OCT_img = img[0, 0, :, :, :]
    OCTA_img = img[0, 1, :, :, :]
    

    thresh_OCT = threshold_otsu(OCT_img.cpu().numpy())
    thresh_OCTA = threshold_otsu(OCTA_img.cpu().numpy())
    
    mask_OCT = OCT_img > thresh_OCT
    mask_OCTA = OCTA_img > thresh_OCTA
    dmin, dmax = 0, img.shape[2]
    
    low_thresh = 0
    high_thresh = torch.numel(mask_OCT)
    while dmax - dmin > target_depth:
        # finding depths with more than sum_thresh pixels for OCT and OCTA image
        sum_thresh = low_thresh + (high_thresh - low_thresh) // 2
        # print(sum_thresh)

        # print(len(torch.where(torch.sum(mask_OCT, axis=(1, 2)) > sum_thresh)[0]))
        depths_OCT = torch.where(torch.sum(mask_OCT, axis=(1, 2)) > sum_thresh)[0]
        depths_OCTA = torch.where(torch.sum(mask_OCTA, axis=(1, 2)) > sum_thresh)[0]
        
        if depths_OCT.shape[0] == 0 or depths_OCTA.shape[0] == 0:
            high_thresh = sum_thresh - 1
            continue
        # dmin_OCT, dmax_OCT = torch.where(torch.sum(mask_OCT, axis=(1, 2)) > sum_thresh)[0][[0, -1]]
        # dmin_OCTA, dmax_OCTA = torch.where(torch.sum(mask_OCTA, axis=(1, 2)) > sum_thresh)[0][[0, -1]]
        
        dmin_OCT, dmax_OCT = depths_OCT[[0, -1]]
        dmin_OCTA, dmax_OCTA = depths_OCTA[[0, -1]]
        
        # take largest range of depths
        dmin = max(dmin_OCT.item(), dmin_OCTA.item())
        dmax = min(dmax_OCT.item(), dmax_OCTA.item())
        
        if dmax - dmin > target_depth:
            low_thresh = sum_thresh + 1
        
        # increase the threshold and run again if the depth is still larger than target depth
        # sum_thresh += 1000

    # expand the depth range to target depth if the depth is smaller than target depth
    roi_depth = dmax - dmin + 1
    if roi_depth <= target_depth:
        offset = (target_depth - roi_depth) // 2
        dmin = max(0, dmin - offset)
        dmax = min(img.shape[2] - 1, dmax + offset) 
        
        current_depth = dmax - dmin + 1
        if  current_depth < target_depth:
            dmax = min(img.shape[2] - 1, dmax + (target_depth - current_depth))
    
    # return the cropped image
    return img[:, :, dmin:dmax + 1, :, :]



# process and concatenate OCT and OCTA image data 
# calculate the projection map
def process_data(OCT_img, OCTA_img, device, input_shape=(128, 256, 256), 
                 roi_target_depth=550, use_proj_map = False, 
                 OCT_proj_map = None, OCTA_proj_map = None):
    OCT_img = torch.from_numpy(OCT_img).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    OCTA_img = torch.from_numpy(OCTA_img).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)
    

    # concatenate the OCT and OCTA images along the channel dimension
    data = torch.cat((OCT_img, OCTA_img), dim=1).to(device)
    
    # get region of interest by cropping depth to target depth
    data = get_ROI(data, roi_target_depth)
    
    # normalize the data along channels
    for i in range(data.shape[1]):
        data[0, i, :, :, :] = normalize(data[0, i, :, :, :])
    
    # resize the data to the input shape
    data = F.interpolate(data, size=input_shape, mode='trilinear', align_corners=True)
    
    # normalize the data after interpolation along channels. 
    for i in range(data.shape[1]):
        data[0, i, :, :, :] = normalize(data[0, i, :, :, :])

    # add Manhattan distance map
    for i in range(data.shape[1]):
        for u in range(data.shape[3]):
                for v in range(data.shape[4]):
                    data[0, i, 0, u, v] = (abs(u - 0.5 * input_shape[1]) + abs(v - 0.5 * input_shape[2]))\
                                                / ( 0.5 * input_shape[1] + 0.5 * input_shape[2])
    
    proj_map = None
    if use_proj_map:
        # generate the projection map using mean along depth if projection map is not provided
        if OCTA_proj_map is None:
            raise ValueError("OCTA projection map must be provided if use_proj_map is True")
        else:
            OCTA_proj_map = torch.from_numpy(OCTA_proj_map).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        
        if OCT_proj_map is None:
            OCT_proj_map = torch.mean(data[0, 0, :, :, :], dim=0).unsqueeze(0).unsqueeze(0)
        else:
            OCT_proj_map = torch.from_numpy(OCT_proj_map).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        
        
        # normalize the projection maps
        OCTA_proj_map = normalize(OCTA_proj_map)
        OCT_proj_map = normalize(OCT_proj_map)
        
        OCT_proj_map = F.interpolate(OCT_proj_map, size=(input_shape[1], input_shape[2]), mode='bilinear', align_corners=True)
        OCTA_proj_map = F.interpolate(OCTA_proj_map, size=(input_shape[1], input_shape[2]), mode='bilinear', align_corners=True)
        
        OCTA_proj_map = normalize(OCTA_proj_map)
        OCT_proj_map = normalize(OCT_proj_map)
        
        # concatenate the projection maps along the channel dimension
        proj_map = torch.cat((OCT_proj_map, OCTA_proj_map), dim=1)
    
    return data, proj_map        