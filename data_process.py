from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSignal, QThread


import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset


from read_write import Read_Data


class Aireadi_Dataset(Dataset):
    def __init__(self, data_map, roi):
        super().__init__()
        self.data_map = data_map
        self.roi = roi
        
        # Build an idx map to map idx to data_map keys
        self.idx_map = dict()
        
        # Convert data_map.keys() to a list
        keys = list(data_map.keys())
        
        # Iterate over the keys and map idx to key
        for i in range(len(keys)):
            self.idx_map[i] = keys[i]
            
        
    def __len__(self):
        return len(self.idx_map)
    
    
    def __getitem__(self, idx):
        oct_fp, octa_fp, enface_fp = self.get_filepath(idx)
        
        OCT_img = Read_Data(oct_fp).get()
        OCTA_img = Read_Data(octa_fp).get()
        
        OCT_img = OCT_img.transpose(1, 0, 2)
        OCTA_img = OCTA_img.transpose(1, 0, 2)
        
        enface_img = Read_Data(enface_fp).get()
        
        data, proj_map = process_data(
            OCT_img=OCT_img, OCTA_img=OCTA_img, roi_target_depth=self.roi, 
            use_proj_map=True, OCTA_proj_map=enface_img
        )
        
        return enface_fp, data, proj_map
    
    
    def get_filepath(self, idx):
        group = self.idx_map[idx]
        
        item = self.data_map[group]
        
        oct_fp = item['oct_path']
        octa_fp = item['octa_path']
        enface_fp = item['enface_path']
        
        return oct_fp, octa_fp, enface_fp
    
    
    
    


class ProcessThread(QThread):
    process = pyqtSignal(int)  # Signal to update progress bar
    work_complete_signal = pyqtSignal()  # Signal when work is finished

    def __init__(self, parent, device):
        super().__init__()
        self.parent = parent
        self.running = True  # Flag to control stopping
        self.device = device
        
        
    
    @torch.no_grad() # Disable gradient calculations
    def run(self):
        
        total_images = len(self.parent.dataset)  # Get total dataset size
        batch_size = min(4, total_images) # make sure batch_size doesn't exceed the number of images
        
        dataloader = DataLoader(
            self.parent.dataset, 
            batch_size=batch_size, 
            pin_memory=True,
            shuffle=False, 
            num_workers=min(4, batch_size), # uses 4 workers maximum
            persistent_workers=True
        )
        
        for batch, (enface_fp, data, proj_map) in enumerate(dataloader):

            # Stop processing if stopped
            if not self.running:
                break  
            
            # half precision
            with autocast(dtype=torch.float16):
                data = data[:,0,...].to(self.device)
                proj_map = proj_map[:,0,...].to(self.device)
                
                cavf_pred, ava_pred, cavf_pred_2D, ava_pred_2D = self.parent.model(data, proj_map)
                cavf_pred_2D = F.softmax(cavf_pred_2D, dim=1).to('cpu').numpy()

            
            # save prediction and model_output to output folder
            for i in range(len(enface_fp)):
                enface_name = os.path.basename(enface_fp[i])
                
                prediction = get_cavf_Sparse_RGBA(cavf_pred_2D[i])
                prediction = cv2.cvtColor(prediction, cv2.COLOR_BGRA2RGBA)
                
                model_output = get_cavf_RGB(cavf_pred_2D[i])
                model_output = cv2.cvtColor(model_output, cv2.COLOR_BGR2RGB)
                
                enface = Read_Data(enface_fp[i]).get()
                overlayed = overlay(enface, prediction)
                
                # write image to folder
                cv2.imwrite(f'{self.parent.output_folder}/{enface_name}_prediction.png', prediction)
                cv2.imwrite(f'{self.parent.output_folder}/{enface_name}_output.png', model_output)
                cv2.imwrite(f'{self.parent.output_folder}/{enface_name}_overlay.png', overlayed)
                
            # Update progress bar
            progress = int((batch + 1)*batch_size / total_images * 100)
            self.process.emit(progress)
        
        self.process.emit(100)
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



# overlay the RGBA vessel map on top of the grayscale enface
def overlay(enface, prediction):
    # convert grayscale to rgb
    enface = cv2.cvtColor(enface, cv2.COLOR_GRAY2RGB)  # Shape: (H, W, 3)
    
    # Resize prediction to match enface size
    prediction = cv2.resize(prediction, (enface.shape[1], enface.shape[0]), interpolation=cv2.INTER_LINEAR)
    #prediction = cv2.cvtColor(prediction, cv2.COLOR_BGRA2RGBA)  # Shape: (H, W, 3)
    
    # Extract the RGB and Alpha channels separately
    prediction_rgb = prediction[:, :, :3]  # RGB part
    alpha = prediction[:, :, 3] / 255.0    # Normalize alpha to range [0,1]
    
    # Blend the images using alpha compositing
    overlay = (1 - alpha[:, :, None]) * enface + alpha[:, :, None] * prediction_rgb
    overlay = overlay.astype(np.uint8)  # Convert back to uint8 format
    
    return overlay



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
def process_data(OCT_img, OCTA_img, input_shape=(128, 256, 256), 
                 roi_target_depth=550, use_proj_map = False, 
                 OCT_proj_map = None, OCTA_proj_map = None):
    
    OCT_img = torch.from_numpy(OCT_img).unsqueeze(0).unsqueeze(0).float()
    OCTA_img = torch.from_numpy(OCTA_img).unsqueeze(0).unsqueeze(0).float()
    

    # concatenate the OCT and OCTA images along the channel dimension
    data = torch.cat((OCT_img, OCTA_img), dim=1)
    
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
            OCTA_proj_map = torch.from_numpy(OCTA_proj_map).unsqueeze(0).unsqueeze(0)
        
        if OCT_proj_map is None:
            OCT_proj_map = torch.mean(data[0, 0, :, :, :], dim=0).unsqueeze(0).unsqueeze(0)
        else:
            OCT_proj_map = torch.from_numpy(OCT_proj_map).unsqueeze(0).unsqueeze(0)
        
        
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