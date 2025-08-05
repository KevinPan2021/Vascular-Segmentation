from PyQt5.QtGui import QImage
from PyQt5.QtCore import pyqtSignal, QThread


import os
import cv2
import numpy as np
import pydicom
import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu
from torch.utils.data import Dataset
from sklearn.neighbors import KNeighborsClassifier

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
        oct_fp, octa_fp, enface_fp, FOV = self.get_filepath(idx)
        
        enface_img = Read_Data(enface_fp).get()
        
        # 3d oct data
        if oct_fp is None:
            OCT_img = np.random.randint(0, 256, (350, 1024, 350), dtype=np.uint8)
        else:
            OCT_img = Read_Data(oct_fp).get()
        
        # 3d octa data
        if octa_fp is None:
            OCTA_img = np.random.randint(0, 256, (350, 1024, 350), dtype=np.uint8)
        else:
            OCTA_img = Read_Data(octa_fp).get()
        
        
        # FOV [1, 0, 0] -> 3mm * 3mm
        # FOV [0, 1, 0] -> 6mm * 6mm
        # FOV [0, 0, 1] -> 12mm * 12mm
        if FOV == '3*3':
            FOV_tensor = torch.tensor([1, 0, 0])
        elif FOV == '6*6':
            FOV_tensor = torch.tensor([0, 1, 0])
        elif FOV == '12*12':
            FOV_tensor = torch.tensor([0, 0, 1])
        
        if octa_fp.endswith('.dcm'):
            octa_dicom = pydicom.dcmread(octa_fp)
            
            pixel_measures = octa_dicom.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            depth_pixel_spacing = pixel_measures.PixelSpacing[1]
            roi = int(2 / depth_pixel_spacing) # target depth of 2mm
        else:
            roi = self.roi
        
        data, proj_map = process_data(
            OCT_img=OCT_img, OCTA_img=OCTA_img, roi_target_depth=roi, 
            use_proj_map=True, OCTA_proj_map=enface_img
        )
        
        return enface_fp, data, proj_map, FOV_tensor
    
    
    def get_filepath(self, idx):
        group = self.idx_map[idx]
        
        item = self.data_map[group]
        
        oct_fp = item['oct_path']
        octa_fp = item['octa_path']
        enface_fp = item['enface_path']
        FOV = item['FOV']
        return oct_fp, octa_fp, enface_fp, FOV
    
    

class ProcessThread(QThread):
    process = pyqtSignal(int)  # Signal to update progress bar
    work_complete_signal = pyqtSignal()  # Signal when work is finished

    def __init__(self, parent=None, device=None):
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
        
        for batch, (enface_fp, data, proj_map, FOV_tensor) in enumerate(dataloader):

            # Stop processing if stopped
            if not self.running:
                break  
            
            # half precision
            with autocast('cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                data = data[:,0,...].to(self.device)
                proj_map = proj_map[:,0,...].to(self.device)
                FOV_tensor = FOV_tensor.to(dtype=data.dtype, device=self.device)
                
                cavf_pred_2D = self.parent.model(data, proj_map, FOV_tensor)[0]
                cavf_pred_2D = F.softmax(cavf_pred_2D, dim=1).to('cpu').numpy()
            
            
            # save prediction and model_output to output folder
            for i in range(len(enface_fp)):
                enface_name = os.path.basename(enface_fp[i])
                
                pred3d_argmax = np.argmax(cavf_pred_2D[i], axis=0)
                
                calc_save_mask_AVA(pred3d_argmax, self.parent.output_folder, enface_name)
                                   

                prediction = get_cavf_Sparse_RGBA(cavf_pred_2D[i])                
                prediction = cv2.cvtColor(prediction, cv2.COLOR_BGRA2RGBA)
                
                model_output = get_cavf_RGB(cavf_pred_2D[i])
                model_output = cv2.cvtColor(model_output, cv2.COLOR_BGR2RGB)
                
                enface = Read_Data(enface_fp[i]).get()
                overlayed = overlay(enface, prediction)
                
                # write image to folder
                filename_no_ext = os.path.splitext(os.path.basename(enface_name))[0]
                output_folder = f'{self.parent.output_folder}/{filename_no_ext}'
                
                if not cv2.imwrite(f'{output_folder}/prediction.png', prediction):
                    raise Exception('file saving error')
                if not cv2.imwrite(f'{output_folder}/output.png', model_output):
                    raise Exception('file saving error')
                if not cv2.imwrite(f'{output_folder}/overlay.png', overlayed):
                    raise Exception('file saving error')
                    
            # Update progress bar
            progress = int((batch + 1)*batch_size / total_images * 100)
            self.process.emit(progress)
        
        self.process.emit(100)
        self.work_complete_signal.emit()  # Signal that work is done


    def stop(self):
        self.running = False  # Set flag to stop processing



# only keep the largest area and fill holes
def FAZ_mask_process(img):
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)

    # Ignore background (label 0), find the label with the largest area
    if num_labels <= 1:
        return np.zeros_like(img)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_component = (labels == largest_label).astype(np.uint8)

    # Fill holes inside the largest component
    # Invert the image to find holes
    holes = cv2.bitwise_not(largest_component * 255)
    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.pointPolygonTest(cnt, (0, 0), False) < 0:  # skip outer contour
            cv2.drawContours(largest_component, [cnt], -1, 1, -1)

    # Convert back to 0/255 format
    result = (largest_component * 255).astype(np.uint8)
    return result
    

# uses KNN to determine the artery and vein zone
def generat_AVA_map(image, vein_mask, artery_mask, FAZ_mask, k=5):
    H, W = image.shape
 
    artery_coords = np.vstack(np.argwhere(image == 2))
    vein_coords = np.vstack(np.argwhere(image == 3))
 
 
    train_data = np.concatenate((artery_coords, vein_coords), axis = 0).astype(np.float32)
 
    labels = np.concatenate( (np.zeros(len(artery_coords)), np.ones(len(vein_coords))) ).astype(np.int32)
 
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, labels)
 
 
    grid_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    grid_coords = np.stack(grid_coords, axis=-1).reshape(-1, 2).astype(np.float32)
    result = knn.predict(grid_coords).reshape(H, W)
    
    # Create an empty RGB image (height, width, 3)
    AVA_map = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    
    # Set blue for 0s
    AVA_map[result == 0] = [100, 100, 255]  # Blue
    
    # Set red for 1s
    AVA_map[result == 1] = [255, 100, 100]  # Red
    
    # overlay on vein / artery
    AVA_map[vein_mask==255] *= np.array([1, 0, 0], dtype=np.uint8)
    AVA_map[artery_mask==255] *= np.array([0, 0, 1], dtype=np.uint8)
    AVA_map[FAZ_mask==255] *= np.array([0, 1, 0], dtype=np.uint8)
    AVA_map[FAZ_mask==255] += np.array([0, 155, 0], dtype=np.uint8)
    
    return AVA_map



def calc_save_mask_AVA(pred3d_argmax, output_path, enface_name, manual=False):
    # capillary, vein, artery, FAZ masks
    capillary_mask = (pred3d_argmax == 1).astype(np.uint8) * 255
    vein_mask = (pred3d_argmax == 3).astype(np.uint8) * 255
    artery_mask = (pred3d_argmax == 2).astype(np.uint8) * 255
    FAZ_mask = (pred3d_argmax == 4).astype(np.uint8) * 255
    FAZ_mask = FAZ_mask_process(FAZ_mask)
    
    # AVA map
    AVA_map = generat_AVA_map(pred3d_argmax, vein_mask, artery_mask, FAZ_mask)
    
    # write image to folder
    filename_no_ext = os.path.splitext(os.path.basename(enface_name))[0]
    output_folder = f'{output_path}/{filename_no_ext}'
    
    suffix = ''
    if manual:
        suffix = '_manual'
        
    if not cv2.imwrite(f'{output_folder}/vein_mask{suffix}.png', vein_mask):
        raise Exception('file saving error')
    if not cv2.imwrite(f'{output_folder}/artery_mask{suffix}.png', artery_mask):
        raise Exception('file saving error')
    if not cv2.imwrite(f'{output_folder}/capillary_mask{suffix}.png', capillary_mask):
        raise Exception('file saving error')
    if not cv2.imwrite(f'{output_folder}/FAZ_mask{suffix}.png', FAZ_mask):
        raise Exception('file saving error')
    if not cv2.imwrite(f'{output_folder}/AVA_map{suffix}.png', AVA_map):
        raise Exception('file saving error')
            
            
            
        
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
    

    thresh_OCT = threshold_otsu(OCT_img.numpy())
    thresh_OCTA = threshold_otsu(OCTA_img.numpy())
    
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